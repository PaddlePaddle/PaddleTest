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



class PrimitiveOp_dc9e2341b6ace7f3c85528fca969390a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2, 3]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_50c2a0658b1824e0277389149f7a0e58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc9e2341b6ace7f3c85528fca969390a
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_09410f01603b0abf026ec76e1af448bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc9e2341b6ace7f3c85528fca969390a
    def get_inputs(self):
        return [
            paddle.uniform([1, 92, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5a85accd097822deb64ef115e86509ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc9e2341b6ace7f3c85528fca969390a
    def get_inputs(self):
        return [
            paddle.uniform([22, 2048, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2d04e22f6593be046b61c2f72469db10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc9e2341b6ace7f3c85528fca969390a
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_44721a76b3edbaab01865aedccb641e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc9e2341b6ace7f3c85528fca969390a
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9eaf7c7fa168d3e53f9315c7b790d2a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc9e2341b6ace7f3c85528fca969390a
    def get_inputs(self):
        return [
            paddle.uniform([10, 336, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_777900657576d7e9e932a993cb87b2f5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9c6e84fa94aff36b902cc81c4858c594(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_777900657576d7e9e932a993cb87b2f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 3549, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_89c0cfc5265c81f3c31fd7bd369b5626(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc9e2341b6ace7f3c85528fca969390a
    def get_inputs(self):
        return [
            paddle.uniform([10, 60, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_0b176bc7a7074e5092e342ecbb8bb391(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0e7f29c154b382e4c5f9d1682252c703(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b176bc7a7074e5092e342ecbb8bb391
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3800, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f5865204ba0ab968b70935028eeb92c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b176bc7a7074e5092e342ecbb8bb391
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[150, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_50d00e3575708548ab78014dfb5c9ca9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc9e2341b6ace7f3c85528fca969390a
    def get_inputs(self):
        return [
            paddle.uniform([145, 336, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_50d00e3575708548ab78014dfb5c9ca9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc9e2341b6ace7f3c85528fca969390a
    def get_inputs(self):
        return [
            paddle.uniform([145, 336, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2b741953fc0545802d417a9932d6d50c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b176bc7a7074e5092e342ecbb8bb391
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[40, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0e7f29c154b382e4c5f9d1682252c703(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b176bc7a7074e5092e342ecbb8bb391
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3800, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_06e5be5b383202141ced5fb540a6e578(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_629012d92bd9d963dbc0664acd91734c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06e5be5b383202141ced5fb540a6e578
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.330821990966797], [2.077406167984009], [1.9619231224060059], [2.031001329421997], [1.9382646083831787], [2.094888687133789], [2.3205490112304688], [2.1239352226257324], [2.0456910133361816], [1.9520155191421509], [2.3500053882598877], [1.9425926208496094], [2.248598098754883], [2.0562705993652344], [1.8699688911437988], [1.959425449371338]], dtype='float32').reshape([16, 1]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8033bd46ee3604295181641799cd9f3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06e5be5b383202141ced5fb540a6e578
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.065321207046509], [2.1375393867492676], [2.2609829902648926], [2.2225561141967773], [2.067194700241089], [2.001950263977051], [2.117380142211914], [2.032435655593872], [1.9074363708496094], [2.158425807952881], [1.868749737739563], [2.2490181922912598], [2.2936840057373047], [2.2391488552093506], [2.0483200550079346], [2.0286765098571777]], dtype='float32').reshape([16, 1]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ac280e06d3e6ba75e25831e14de31129(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc9e2341b6ace7f3c85528fca969390a
    def get_inputs(self):
        return [
            paddle.uniform([145, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_fde6264d9ca7093190d60e839dda71d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_777900657576d7e9e932a993cb87b2f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 7581, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_713d9cf7bce2e6fe1a984f6d97d16297(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2a547b12d60200175c68b821da47d396(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_713d9cf7bce2e6fe1a984f6d97d16297
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 18, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2a547b12d60200175c68b821da47d396(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_713d9cf7bce2e6fe1a984f6d97d16297
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 18, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_ee3ce259b2a3ddaed918b21535078a2a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_88b1312d3f3456fd9dcb9390c82aa550(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee3ce259b2a3ddaed918b21535078a2a
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 66, 130], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_f3a86fe2cac3346d5282bcadb8faa108(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_777900657576d7e9e932a993cb87b2f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4725, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_46d082d660564190158bc00eac2f58ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc9e2341b6ace7f3c85528fca969390a
    def get_inputs(self):
        return [
            paddle.uniform([22, 60, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b343f372fbe02d2ac2d20c0e9609f814(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc9e2341b6ace7f3c85528fca969390a
    def get_inputs(self):
        return [
            paddle.uniform([1, 872, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_ba8b329fbcd2f1625386e7d84389b057(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_03bf280c40074fdef157bd1d0c80d81d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba8b329fbcd2f1625386e7d84389b057
    def get_inputs(self):
        return [
            paddle.uniform([1762, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_03bf280c40074fdef157bd1d0c80d81d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba8b329fbcd2f1625386e7d84389b057
    def get_inputs(self):
        return [
            paddle.uniform([1762, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_1dc0076ca43550028f8b6899dae236c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_777900657576d7e9e932a993cb87b2f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8400, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_71b797d9b4af0663609246ff9427f2f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc9e2341b6ace7f3c85528fca969390a
    def get_inputs(self):
        return [
            paddle.uniform([171, 336, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class PrimitiveOp_20b196684d5a9262d88a94fe91836e2f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [2]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_45e552f20955ddb1815d1b3c7e36b463(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20b196684d5a9262d88a94fe91836e2f
    def get_inputs(self):
        return [
            paddle.uniform([43, 768, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9c6e84fa94aff36b902cc81c4858c594(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_777900657576d7e9e932a993cb87b2f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 3549, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6aec59dc6f580f7adc3b5269a62bbe4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc9e2341b6ace7f3c85528fca969390a
    def get_inputs(self):
        return [
            paddle.uniform([10, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_bafb7140e883b23efcc734731d9da984(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba8b329fbcd2f1625386e7d84389b057
    def get_inputs(self):
        return [
            paddle.uniform([5522, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bafb7140e883b23efcc734731d9da984(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba8b329fbcd2f1625386e7d84389b057
    def get_inputs(self):
        return [
            paddle.uniform([5522, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7b1ce5fa1fabdff5ce2da66df18702e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06e5be5b383202141ced5fb540a6e578
    def get_inputs(self):
        return [
            paddle.uniform([36, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7b1ce5fa1fabdff5ce2da66df18702e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06e5be5b383202141ced5fb540a6e578
    def get_inputs(self):
        return [
            paddle.uniform([36, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_37aba99f3b7a1e038ff5a52e969912db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc9e2341b6ace7f3c85528fca969390a
    def get_inputs(self):
        return [
            paddle.uniform([43, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9eaf7c7fa168d3e53f9315c7b790d2a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc9e2341b6ace7f3c85528fca969390a
    def get_inputs(self):
        return [
            paddle.uniform([10, 336, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5f9dab5b375b278f3b7c6636842f9f2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b176bc7a7074e5092e342ecbb8bb391
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[15200, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5f9dab5b375b278f3b7c6636842f9f2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b176bc7a7074e5092e342ecbb8bb391
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[15200, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8afb2712c08c1417d5f59050badb165d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc9e2341b6ace7f3c85528fca969390a
    def get_inputs(self):
        return [
            paddle.uniform([10, 36, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4a1518e096b8dac596ebb0d6a5a83291(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc9e2341b6ace7f3c85528fca969390a
    def get_inputs(self):
        return [
            paddle.uniform([43, 1280, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2e7b73b78a5d0255596be00fb0dbd344(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc9e2341b6ace7f3c85528fca969390a
    def get_inputs(self):
        return [
            paddle.uniform([10, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_657955bd229e3d8b33417a7fb1ec2fcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc9e2341b6ace7f3c85528fca969390a
    def get_inputs(self):
        return [
            paddle.uniform([10, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0956b34a2000e870033ad9d87e364dc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba8b329fbcd2f1625386e7d84389b057
    def get_inputs(self):
        return [
            paddle.uniform([1760, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0956b34a2000e870033ad9d87e364dc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba8b329fbcd2f1625386e7d84389b057
    def get_inputs(self):
        return [
            paddle.uniform([1760, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_755256a1ba84681878e2a6e4ac31ec38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc9e2341b6ace7f3c85528fca969390a
    def get_inputs(self):
        return [
            paddle.uniform([22, 336, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_7a887f2ad7aba8a4942ccd6bbcf76337(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_777900657576d7e9e932a993cb87b2f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4116, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a714798365a39b0e508dd900d88bbe1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc9e2341b6ace7f3c85528fca969390a
    def get_inputs(self):
        return [
            paddle.uniform([171, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_71b797d9b4af0663609246ff9427f2f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc9e2341b6ace7f3c85528fca969390a
    def get_inputs(self):
        return [
            paddle.uniform([171, 336, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a6df72c332bed9214a57e3cdb653306a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc9e2341b6ace7f3c85528fca969390a
    def get_inputs(self):
        return [
            paddle.uniform([22, 1536, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_03123fb5ed3108210f2e1efdfc14f341(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06e5be5b383202141ced5fb540a6e578
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.231990337371826], [1.9275521039962769], [2.2786357402801514], [2.054826259613037], [2.0829288959503174], [1.9492191076278687], [2.174846887588501], [2.0490922927856445], [2.0114030838012695], [2.1977427005767822], [2.3248181343078613], [2.2420318126678467], [1.9566590785980225], [1.9651343822479248], [2.127338409423828], [2.0439677238464355], [1.9792596101760864], [2.2133615016937256], [1.898798942565918], [2.011535406112671], [2.0508549213409424], [2.111821174621582], [2.070446491241455], [2.233060598373413]], dtype='float32').reshape([24, 1]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5c0ae6d534265fc62d88bf0c9a964567(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06e5be5b383202141ced5fb540a6e578
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.24531888961792], [2.0377137660980225], [2.2062227725982666], [1.7972369194030762], [2.2450249195098877], [2.106555461883545], [2.0480704307556152], [1.937278151512146], [2.070096254348755], [1.8892933130264282], [2.016486167907715], [2.1491715908050537], [2.2348837852478027], [2.034191608428955], [2.3335111141204834], [2.2069406509399414], [2.0489346981048584], [1.8884472846984863], [1.9661571979522705], [1.8743488788604736], [1.9377069473266602], [1.9976824522018433], [1.8631433248519897], [1.9819202423095703]], dtype='float32').reshape([24, 1]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_0004e9213c74f18d5f42c14950df62c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc9e2341b6ace7f3c85528fca969390a
    def get_inputs(self):
        return [
            paddle.uniform([171, 60, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_84fb3fe4474937317b3294c1eb34c2e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_777900657576d7e9e932a993cb87b2f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 6069, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_345e1c4588516fa62e082364aacdf039(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba8b329fbcd2f1625386e7d84389b057
    def get_inputs(self):
        return [
            paddle.uniform([1522, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_345e1c4588516fa62e082364aacdf039(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba8b329fbcd2f1625386e7d84389b057
    def get_inputs(self):
        return [
            paddle.uniform([1522, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ba8dc51897f4780aac44d0ab1d2bb0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc9e2341b6ace7f3c85528fca969390a
    def get_inputs(self):
        return [
            paddle.uniform([22, 240, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ba361cd536816d4e5b4cef496f34323a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc9e2341b6ace7f3c85528fca969390a
    def get_inputs(self):
        return [
            paddle.uniform([10, 1536, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_09301a9cb6928cd47131989654e3d126(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06e5be5b383202141ced5fb540a6e578
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.0250163078308105], [1.9384316205978394], [1.881026029586792], [2.1979641914367676]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_28a0570be4ed7e08ad121ec04775d3ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06e5be5b383202141ced5fb540a6e578
    def get_inputs(self):
        return [
            paddle.to_tensor([[1.9154152870178223], [2.065770387649536], [2.0589423179626465], [1.881870150566101]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ae303e9514454bc916599b3c5734d1e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee3ce259b2a3ddaed918b21535078a2a
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 70, 134], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_675f46e8f17768cc62dc66dd636279fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee3ce259b2a3ddaed918b21535078a2a
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 104, 101], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_58763d386914efb4ef748656c49fb173(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b176bc7a7074e5092e342ecbb8bb391
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2204, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_170d49f25b8918b3f69632c3101bdc3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc9e2341b6ace7f3c85528fca969390a
    def get_inputs(self):
        return [
            paddle.uniform([22, 36, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_efb77cc32cae3e0c801c454654416ad7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee3ce259b2a3ddaed918b21535078a2a
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 68, 132], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bc39ab39a0445673393c059ee0516d41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc9e2341b6ace7f3c85528fca969390a
    def get_inputs(self):
        return [
            paddle.uniform([11, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_05e6f9c82d939051b4c765a9d6aa1c3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc9e2341b6ace7f3c85528fca969390a
    def get_inputs(self):
        return [
            paddle.uniform([145, 60, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_9e2be14b7ef3fe6e2a4afab3f83fb202(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_713d9cf7bce2e6fe1a984f6d97d16297
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 36, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_9e2be14b7ef3fe6e2a4afab3f83fb202(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_713d9cf7bce2e6fe1a984f6d97d16297
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 36, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b3d3c3e546e0e7638b042640dfc4d846(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b176bc7a7074e5092e342ecbb8bb391
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[70, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_75f96cc1d8926b0c2dc0d4247b756940(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc9e2341b6ace7f3c85528fca969390a
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d92d05ce0893303b95d5f178d69691f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b176bc7a7074e5092e342ecbb8bb391
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[551, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_755256a1ba84681878e2a6e4ac31ec38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc9e2341b6ace7f3c85528fca969390a
    def get_inputs(self):
        return [
            paddle.uniform([22, 336, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d41977da2d5cb93a588ecae103dc5d03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b176bc7a7074e5092e342ecbb8bb391
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[247, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bbf17cfd54b540ac15cb60392ee00b26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc9e2341b6ace7f3c85528fca969390a
    def get_inputs(self):
        return [
            paddle.uniform([10, 2048, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_a1e6209bd4847b46169c1697f45b1501(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b176bc7a7074e5092e342ecbb8bb391
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[950, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8931fd6fa2c126d3366212f6dd0705a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba8b329fbcd2f1625386e7d84389b057
    def get_inputs(self):
        return [
            paddle.uniform([2074, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8931fd6fa2c126d3366212f6dd0705a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba8b329fbcd2f1625386e7d84389b057
    def get_inputs(self):
        return [
            paddle.uniform([2074, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_acc06ef054b6ce1c26de798d2d229511(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b176bc7a7074e5092e342ecbb8bb391
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[8816, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_37de1dff3c15688e9881dcde206ae706(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba8b329fbcd2f1625386e7d84389b057
    def get_inputs(self):
        return [
            paddle.uniform([4734, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_37de1dff3c15688e9881dcde206ae706(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba8b329fbcd2f1625386e7d84389b057
    def get_inputs(self):
        return [
            paddle.uniform([4734, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_7864c1428fac938768669353a64e0542(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20b196684d5a9262d88a94fe91836e2f
    def get_inputs(self):
        return [
            paddle.uniform([10, 96, 1, 40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2f6d2a9ae695a6e7db23d77c928da22b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba8b329fbcd2f1625386e7d84389b057
    def get_inputs(self):
        return [
            paddle.uniform([1, 2434, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_5e1431028c5581d7675496036491b51b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_afad4e01fa637835f804838dc93f88bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e1431028c5581d7675496036491b51b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 2434, 1], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d7efa9a50fe2b30c66137475799c6260(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba8b329fbcd2f1625386e7d84389b057
    def get_inputs(self):
        return [
            paddle.uniform([1074, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d7efa9a50fe2b30c66137475799c6260(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba8b329fbcd2f1625386e7d84389b057
    def get_inputs(self):
        return [
            paddle.uniform([1074, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ac1207700f807063863fdfdbdb032a58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_777900657576d7e9e932a993cb87b2f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 9261, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ae563ec21dc609384d2d690b062eab0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20b196684d5a9262d88a94fe91836e2f
    def get_inputs(self):
        return [
            paddle.uniform([11, 768, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8d14ba0d23dda7ebb8eccf37ae33c8f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee3ce259b2a3ddaed918b21535078a2a
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_a263f96f115bb9957a4296b0d3d247e0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6855bff0aa1e852777dc8fef067cde9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a263f96f115bb9957a4296b0d3d247e0
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_55d573e7d91d251e04b073e821a0c3ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba8b329fbcd2f1625386e7d84389b057
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2686c23c5b9b0ac0db0efef319aacd4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_777900657576d7e9e932a993cb87b2f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2100, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_784941def2bae63ac9e65057b9285cfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc9e2341b6ace7f3c85528fca969390a
    def get_inputs(self):
        return [
            paddle.uniform([1, 1248, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_8bcdbbfd1dcf3fa5f6f3b1fbe7a2159f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc9e2341b6ace7f3c85528fca969390a
    def get_inputs(self):
        return [
            paddle.uniform([171, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_cfecf4a864e3db15c4fe4bf76d3d276d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc9e2341b6ace7f3c85528fca969390a
    def get_inputs(self):
        return [
            paddle.uniform([145, 36, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_5acd9f8b109159bb1ba640781b3815a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_713d9cf7bce2e6fe1a984f6d97d16297
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 9, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_5acd9f8b109159bb1ba640781b3815a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_713d9cf7bce2e6fe1a984f6d97d16297
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 9, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ce6b950a0cc5f3660da15da38a6beffc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba8b329fbcd2f1625386e7d84389b057
    def get_inputs(self):
        return [
            paddle.uniform([2332, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ce6b950a0cc5f3660da15da38a6beffc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba8b329fbcd2f1625386e7d84389b057
    def get_inputs(self):
        return [
            paddle.uniform([2332, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_26305563fc0b71885f87e94b391ad459(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_713d9cf7bce2e6fe1a984f6d97d16297
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 96, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_26305563fc0b71885f87e94b391ad459(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_713d9cf7bce2e6fe1a984f6d97d16297
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 96, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2ce8a7d3a334a21929658ff0bbc4ad21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba8b329fbcd2f1625386e7d84389b057
    def get_inputs(self):
        return [
            paddle.uniform([3051, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_2ce8a7d3a334a21929658ff0bbc4ad21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba8b329fbcd2f1625386e7d84389b057
    def get_inputs(self):
        return [
            paddle.uniform([3051, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8060d8ca6831da675a23cc23aa443ead(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba8b329fbcd2f1625386e7d84389b057
    def get_inputs(self):
        return [
            paddle.uniform([3870, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8060d8ca6831da675a23cc23aa443ead(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba8b329fbcd2f1625386e7d84389b057
    def get_inputs(self):
        return [
            paddle.uniform([3870, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b381137f85f830f801e93486261a774b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_713d9cf7bce2e6fe1a984f6d97d16297
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 24, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b381137f85f830f801e93486261a774b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_713d9cf7bce2e6fe1a984f6d97d16297
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 24, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4bfb494ed0149e1f555e74c36fcda9ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc9e2341b6ace7f3c85528fca969390a
    def get_inputs(self):
        return [
            paddle.uniform([1, 156, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_319b4bf60be0f2958864f702691f9c53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_713d9cf7bce2e6fe1a984f6d97d16297
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 48, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_319b4bf60be0f2958864f702691f9c53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_713d9cf7bce2e6fe1a984f6d97d16297
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 48, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_fd90a96709236f212f6a4c5d3ed86f1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_777900657576d7e9e932a993cb87b2f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 11109, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_b343f372fbe02d2ac2d20c0e9609f814(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc9e2341b6ace7f3c85528fca969390a
    def get_inputs(self):
        return [
            paddle.uniform([1, 872, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_6e23f1c6812315ffc24fbdfbd2262b55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc9e2341b6ace7f3c85528fca969390a
    def get_inputs(self):
        return [
            paddle.uniform([22, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_2300b8d6a8167bc294844b8c14bb108e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc9e2341b6ace7f3c85528fca969390a
    def get_inputs(self):
        return [
            paddle.uniform([145, 480, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_ae50e52fee77eb717645ffaaecec2d49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20b196684d5a9262d88a94fe91836e2f
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 1, 25], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_bce92d9a8799778d91c6a355f36eabd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc9e2341b6ace7f3c85528fca969390a
    def get_inputs(self):
        return [
            paddle.uniform([171, 36, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_d2492ce4979bc7106a48aca5153a0a09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc9e2341b6ace7f3c85528fca969390a
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b1394fb9bc9a8b3c4d2843680e8d4eb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06e5be5b383202141ced5fb540a6e578
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.039003610610962], [1.9042551517486572], [2.0163393020629883], [2.1727426052093506], [2.382507801055908], [1.9616048336029053], [1.904478907585144], [1.9909439086914062], [1.9326372146606445], [1.9643150568008423], [2.243607997894287], [1.8741915225982666], [2.2294907569885254], [2.278374671936035], [1.9657965898513794], [2.1199238300323486], [1.8401740789413452], [1.9877119064331055], [2.284308433532715], [1.9237463474273682]], dtype='float32').reshape([20, 1]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d1eab9ae33296cd482f0bafc01b94046(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06e5be5b383202141ced5fb540a6e578
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.1977758407592773], [2.1340818405151367], [2.124262571334839], [1.972059965133667], [2.1026735305786133], [1.9690611362457275], [2.1583495140075684], [2.007345676422119], [2.2774620056152344], [1.8715198040008545], [1.9197783470153809], [1.8987236022949219], [2.051024913787842], [2.001521348953247], [1.9752699136734009], [2.252300977706909], [2.2869536876678467], [2.019422769546509], [1.9557440280914307], [2.145960569381714]], dtype='float32').reshape([20, 1]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d41977da2d5cb93a588ecae103dc5d03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b176bc7a7074e5092e342ecbb8bb391
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[247, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_75f96cc1d8926b0c2dc0d4247b756940(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc9e2341b6ace7f3c85528fca969390a
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_0e7f29c154b382e4c5f9d1682252c703(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b176bc7a7074e5092e342ecbb8bb391
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3800, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_cf4929b4dd10dcd1dfd3520385e19cf6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba8b329fbcd2f1625386e7d84389b057
    def get_inputs(self):
        return [
            paddle.uniform([1, 8732, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_8231a924815c14c8f1c613ce8b478a56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e1431028c5581d7675496036491b51b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 8732, 1], dtype='int64'),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_a1e6209bd4847b46169c1697f45b1501(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b176bc7a7074e5092e342ecbb8bb391
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[950, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c9243b05d3c430854d4742c1168f3a59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba8b329fbcd2f1625386e7d84389b057
    def get_inputs(self):
        return [
            paddle.uniform([2111, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c9243b05d3c430854d4742c1168f3a59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba8b329fbcd2f1625386e7d84389b057
    def get_inputs(self):
        return [
            paddle.uniform([2111, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_ced3b436d65487e62504342c0711a509(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc9e2341b6ace7f3c85528fca969390a
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_b3d3c3e546e0e7638b042640dfc4d846(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b176bc7a7074e5092e342ecbb8bb391
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[70, 1], dtype='int64'),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_d42b0803aa4ba6de7d6aef8e7ba07bad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_777900657576d7e9e932a993cb87b2f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 3024, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4e5f812ca26935d4d47e614047b501a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc9e2341b6ace7f3c85528fca969390a
    def get_inputs(self):
        return [
            paddle.uniform([11, 1280, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_4638fe38f083a980401783cbb3dd65a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba8b329fbcd2f1625386e7d84389b057
    def get_inputs(self):
        return [
            paddle.uniform([4141, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_4638fe38f083a980401783cbb3dd65a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba8b329fbcd2f1625386e7d84389b057
    def get_inputs(self):
        return [
            paddle.uniform([4141, 4, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_63138176202411d6abff1f56870ff3eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc9e2341b6ace7f3c85528fca969390a
    def get_inputs(self):
        return [
            paddle.uniform([1, 624, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


class TestPrimitiveOp_db3b69aad2f12380cb1cf94923b12571(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a263f96f115bb9957a4296b0d3d247e0
    def get_inputs(self):
        return [
            paddle.uniform([10, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_c203501977259c751e5782b943ce998a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba8b329fbcd2f1625386e7d84389b057
    def get_inputs(self):
        return [
            paddle.uniform([10, 1000, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()