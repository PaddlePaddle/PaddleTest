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



class PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 1, paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_f684bc3beb79b493a511c02af321449e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 2, paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1a8204a39f0d0f1149f9bfcde90cd701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f684bc3beb79b493a511c02af321449e
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 1, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_76ace68dab376620e9fb04790a7cf46b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], -1, paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_19ebdd27d8f55d774602a71fb3a20612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76ace68dab376620e9fb04790a7cf46b
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], -1, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_22b303d42b77269d95a42e7eeb558e32(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 36, paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7f7f02bc9164225f4044f4708bf7d244(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22b303d42b77269d95a42e7eeb558e32
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_c5dcae5f2ca7064cb4d3ed64672f0e29(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 24, paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_384f2942846c6c6e20e71dcf0c0f1384(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5dcae5f2ca7064cb4d3ed64672f0e29
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 1, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_59d428e94533601bc16671b552584b86(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 24, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ab6e3f824e07eaa14216b70889b9b4c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59d428e94533601bc16671b552584b86
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_606892bbebc398b07aa0b00da1b29a42(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 36, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_25b952fe4398a4e09ea4da83f66a99f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_606892bbebc398b07aa0b00da1b29a42
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_4155a712ef006bed9a4cd9a779378da4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 0, paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_ee40ea0e78a7cdebaef17925b3c1562a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 300, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_19d0e3cbe63b84c62cee78b7412ba7f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee40ea0e78a7cdebaef17925b3c1562a
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_869f62127d7a1b1c07ad3726a993b605(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 0, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_15006aad1c5611252c9215622d98664a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 0.176777, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c42ea924c62569f1674b9193c5a808bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15006aad1c5611252c9215622d98664a
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_2206545d093007dffd5065cf3c441627(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 0.125, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2c2593aa06a2163b3136a90856e2581c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2206545d093007dffd5065cf3c441627
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_3eb7da9c0e577dcd8562d57bcb61f180(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 1.17647, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4e46abfd416b9997a1a8a2d8db18737e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3eb7da9c0e577dcd8562d57bcb61f180
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_7986a3d6d389630a91a1cc7dbb92b680(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 0.5, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0168b1e8a6ae8a04d43625ae2f3d2d52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7986a3d6d389630a91a1cc7dbb92b680
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_df95f02ddb4bb3e8a56884440e59e08f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 1.14286, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ddde7cefa542fb6dd073cc665e63e2d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df95f02ddb4bb3e8a56884440e59e08f
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_d2fced7ad1d49ed24bed710a63e84ce2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], -1e+10, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c14218e5e4c219495d91681290328607(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2fced7ad1d49ed24bed710a63e84ce2
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_8ae90ea8689aa410ca7d2826b0120d18(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 4.13517, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_06354aeab0d48eb7c15d22c6b59949ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ae90ea8689aa410ca7d2826b0120d18
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_0168b1e8a6ae8a04d43625ae2f3d2d52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7986a3d6d389630a91a1cc7dbb92b680
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_8e7f8c962c4c4f3f50acb5310cf1334e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 64, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a87152b3cd089e715310903b248e3d0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e7f8c962c4c4f3f50acb5310cf1334e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 3.40282e+38, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_95109bd6b94b41007c0e1f63000f70f3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 1, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a558e0793c31ff91eacbba387bdfe25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95109bd6b94b41007c0e1f63000f70f3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_9b77676a4bf5b87dd75109a82f0f76c1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], -1, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f633efdab7ab28ab6c0765b90d0bc2de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b77676a4bf5b87dd75109a82f0f76c1
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_6a4f96fdfa86d19cdd1c558cd6713c0e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 4, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8c8afa339da46c3c4e981fe750f61605(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a4f96fdfa86d19cdd1c558cd6713c0e
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_a4c31049513dc61e0ef38bf74bcd91b4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 19, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b0a5f722ee7e58afbe2e2f5394154f9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c31049513dc61e0ef38bf74bcd91b4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_1a8204a39f0d0f1149f9bfcde90cd701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f684bc3beb79b493a511c02af321449e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_19ebdd27d8f55d774602a71fb3a20612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76ace68dab376620e9fb04790a7cf46b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_8a5da53036182f346c07ea5fc3ed8632(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 0, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_827d102786d013b40bdd26b7ef63298e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a5da53036182f346c07ea5fc3ed8632
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_7209f99ccd37b7e49637ebaa32a52f80(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 1, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4e8fa2e227d494a3f6a8522b5905f72f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7209f99ccd37b7e49637ebaa32a52f80
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_4143be70b6a83709b075d6203f2a5577(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 32, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_68a825a78a12c2b31f7340d146873488(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4143be70b6a83709b075d6203f2a5577
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_1e5be3a4d141b05ca8e3c269ba9e2133(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 16, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_553d5f29dc5959ffeb18466b22d6ff2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e5be3a4d141b05ca8e3c269ba9e2133
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_364e0a5000b30cbe7447f3a7b48d65aa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 8, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2309a2f8454e3f25ed99cebb50f7b2d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_364e0a5000b30cbe7447f3a7b48d65aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_f44c6866020e1c0100d135816a48333c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 80, paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a0ac03445350f26835d65b207edb70ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f44c6866020e1c0100d135816a48333c
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a558e0793c31ff91eacbba387bdfe25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95109bd6b94b41007c0e1f63000f70f3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_993f0e791c7377a5b0e132b3805d13e2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 0.1, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_039fc396b102472fb4a6e5a0fcecc254(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_993f0e791c7377a5b0e132b3805d13e2
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_e9d0e23085c4d0c96e3e881d0902c120(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 0.2, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_485dde1c74b3455de459aa6fbb8feed3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9d0e23085c4d0c96e3e881d0902c120
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_2fd28fc70cde4a4f1950368004d1221b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 100, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_897010487cc007ef1e9b64a8493d0e12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2fd28fc70cde4a4f1950368004d1221b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_900902bf4b6bf411697f3e4135095bdc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 0.111111, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_414cbe2c49ddbc62cc270018556d3439(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_900902bf4b6bf411697f3e4135095bdc
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_0168b1e8a6ae8a04d43625ae2f3d2d52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7986a3d6d389630a91a1cc7dbb92b680
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_5cf049d907e5d6670f2a060e1085f8d1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 9, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_509ed8321457ade6de3d772f1fe18502(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5cf049d907e5d6670f2a060e1085f8d1
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a558e0793c31ff91eacbba387bdfe25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95109bd6b94b41007c0e1f63000f70f3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_0b9bd69b9f40c9f943187eb6d703f689(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 2, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a591d97808b801aa8b83999c48d59b7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b9bd69b9f40c9f943187eb6d703f689
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_7d22c0f2354835635eaf0fa0fd6c227f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 20, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b578e6231c9ab10277dee9851a40ac93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d22c0f2354835635eaf0fa0fd6c227f
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_cc7e398a39cf700afdb757f289c34009(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 40, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_308150ae4577ed08ed610f4be89d45dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc7e398a39cf700afdb757f289c34009
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_54a95fdf6878864a27a5fdbcb8e7ef25(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 80, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_65ef72be708bae8104babbffa128a769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54a95fdf6878864a27a5fdbcb8e7ef25
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_f633efdab7ab28ab6c0765b90d0bc2de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b77676a4bf5b87dd75109a82f0f76c1
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_56e00f90b1ee2c2ad63326a5d6bb69ed(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], -2, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_375969ca7497e3f3e0a6e699730622b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56e00f90b1ee2c2ad63326a5d6bb69ed
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_3447c8d1bdff45772f6c683747ac318b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 0, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2e624ca7b64813c0ca2b1d5c90749ad6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3447c8d1bdff45772f6c683747ac318b
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_96190683c5a4edff8d8977d736e15e7d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 20, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e674bd5969b49291fe1545dcad38919e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96190683c5a4edff8d8977d736e15e7d
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_cfe5d6500f99a977d9da41a9a7128a18(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 21, paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4272819eb3706703baa0eaf500a2fa16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cfe5d6500f99a977d9da41a9a7128a18
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_68134d372e1c79bfd8c2095d4516b8f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1, 500, 1], 0, paddle.int64, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8517bafd6e021a5b61451455f6f231dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_68134d372e1c79bfd8c2095d4516b8f9
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_1a8204a39f0d0f1149f9bfcde90cd701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f684bc3beb79b493a511c02af321449e
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_7f0dff07192990077d6fb94aa22aa2f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 0, paddle.int32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_691e735f0b5f309d6a88dd59b3ae1054(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f0dff07192990077d6fb94aa22aa2f9
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_9ae06a6a28edd1d4ef7daa2193955670(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 128, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_00b5fe9b425704b6a267f9fa46c51e6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ae06a6a28edd1d4ef7daa2193955670
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_414cbe2c49ddbc62cc270018556d3439(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_900902bf4b6bf411697f3e4135095bdc
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_0168b1e8a6ae8a04d43625ae2f3d2d52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7986a3d6d389630a91a1cc7dbb92b680
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_509ed8321457ade6de3d772f1fe18502(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5cf049d907e5d6670f2a060e1085f8d1
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_887720c8954f34c5106fb9e0d46923b5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 96, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_99656affb78e594a0c0f8976e65efdd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_887720c8954f34c5106fb9e0d46923b5
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2309a2f8454e3f25ed99cebb50f7b2d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_364e0a5000b30cbe7447f3a7b48d65aa
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_523c48360cb12a7eb3ebcec2fcba0b01(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([9216, 1], 8, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_32e8ce6b03b0189c40957ab944723849(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_523c48360cb12a7eb3ebcec2fcba0b01
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_132979666a93fd42025b455a4e18a3d0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 48, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e9366e4242b5cd57f878cb674ac7f67e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_132979666a93fd42025b455a4e18a3d0
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_553d5f29dc5959ffeb18466b22d6ff2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e5be3a4d141b05ca8e3c269ba9e2133
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_76255936bde905a22751af5047182f31(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([2304, 1], 16, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_13d87867caef3a468f5c1c058788b0b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76255936bde905a22751af5047182f31
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_c636f5fff1fc99c1cfb715dab7b2c2d2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 24, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b47a884d7150d786fb53dafcad45680c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c636f5fff1fc99c1cfb715dab7b2c2d2
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_68a825a78a12c2b31f7340d146873488(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4143be70b6a83709b075d6203f2a5577
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_22f3c3a2b16a50d0d14831da02dfb33d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([576, 1], 32, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9aaf014d4a146287310cb76135c5a72e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22f3c3a2b16a50d0d14831da02dfb33d
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_1a8204a39f0d0f1149f9bfcde90cd701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f684bc3beb79b493a511c02af321449e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_0168b1e8a6ae8a04d43625ae2f3d2d52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7986a3d6d389630a91a1cc7dbb92b680
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_19ebdd27d8f55d774602a71fb3a20612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76ace68dab376620e9fb04790a7cf46b
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_e64194509319894b799f6df30002a5cb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 7, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aabcd7dcd2477b681b01226445ecb077(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64194509319894b799f6df30002a5cb
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_8aab409109d28df903667e0731215e8b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 768, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8bb29e2528dbddef14fa8be190b7ec6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8aab409109d28df903667e0731215e8b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_0d4fa0aca5c08978162ef63efa409901(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([3800, 80], 0, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6db247c81155d36cf7ab09cc14f3cd8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d4fa0aca5c08978162ef63efa409901
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a558e0793c31ff91eacbba387bdfe25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95109bd6b94b41007c0e1f63000f70f3
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_5e0c50876d98164601ccb8a967b37066(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 0, paddle.int64, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3926a2dc52ea7242a289193e0a6264fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e0c50876d98164601ccb8a967b37066
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_79cd1c1d54ebd9f9c9bdac524eaa7920(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 80, paddle.int64, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7728bdc486306943077ea31df1cefcfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79cd1c1d54ebd9f9c9bdac524eaa7920
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_91744f754c892adcead2511f1f945179(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 2.5, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6a8b89d6ba67997a4f45599dfb6487fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91744f754c892adcead2511f1f945179
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a558e0793c31ff91eacbba387bdfe25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95109bd6b94b41007c0e1f63000f70f3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_1e006158c8e80926f5ba717c1a521302(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 3, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c7b9494b29d2992c39fdd2190f2bc4ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e006158c8e80926f5ba717c1a521302
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_81d652c8795e02c18aa08623c66c64dc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 20, paddle.int64, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2c3bc31a54043a4237c308e1e0c5242c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81d652c8795e02c18aa08623c66c64dc
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_70c7f3c9bcf992d7744af26d83e7c888(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([150, 80], 0, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_99c0f21f57cba7f393d73053ca7c8e46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70c7f3c9bcf992d7744af26d83e7c888
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a558e0793c31ff91eacbba387bdfe25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95109bd6b94b41007c0e1f63000f70f3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_3926a2dc52ea7242a289193e0a6264fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e0c50876d98164601ccb8a967b37066
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7728bdc486306943077ea31df1cefcfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79cd1c1d54ebd9f9c9bdac524eaa7920
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a558e0793c31ff91eacbba387bdfe25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95109bd6b94b41007c0e1f63000f70f3
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_d8fd2faeade559d77a96aa489df0cf0c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 0.00390625, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_572c650ff72ba7e0d118751410bf2ca9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8fd2faeade559d77a96aa489df0cf0c
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_8aab910f89c1b1f773daf7ae13e0507e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 1, paddle.int32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_59884f606a020622bdd05fec914981cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8aab910f89c1b1f773daf7ae13e0507e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_691e735f0b5f309d6a88dd59b3ae1054(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f0dff07192990077d6fb94aa22aa2f9
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_d8b26ad00a514dfeff3b7c530feb5e6e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 256, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e13f875b3cb7da10ff338fc779ce8158(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8b26ad00a514dfeff3b7c530feb5e6e
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_30eac4ed0786aaf3201d15393fda7bec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 21, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4b361d1c0412f28a91ca5a74ad955f46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30eac4ed0786aaf3201d15393fda7bec
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8517bafd6e021a5b61451455f6f231dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_68134d372e1c79bfd8c2095d4516b8f9
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_1a8204a39f0d0f1149f9bfcde90cd701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f684bc3beb79b493a511c02af321449e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_691e735f0b5f309d6a88dd59b3ae1054(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f0dff07192990077d6fb94aa22aa2f9
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_59884f606a020622bdd05fec914981cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8aab910f89c1b1f773daf7ae13e0507e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_691e735f0b5f309d6a88dd59b3ae1054(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f0dff07192990077d6fb94aa22aa2f9
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_1a8204a39f0d0f1149f9bfcde90cd701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f684bc3beb79b493a511c02af321449e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_a3d93c03bbda76945e5889f994faa533(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], -6, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_88c164d1c1c69371788801ba355642f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3d93c03bbda76945e5889f994faa533
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_5fdad3ece117b57f1a8f79391050f6fd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 0.0833333, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dc037143006bc59a51d336e0c77a92e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fdad3ece117b57f1a8f79391050f6fd
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_0ab24cfbb13fe7a706d01982fb678dd3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 6.28319, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2ea5ed02d2ded07a1435b86433f23cd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ab24cfbb13fe7a706d01982fb678dd3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_a4936ab50077760dead200ad28e5aab0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 8, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4d7b7244bc884b86649fb51f2a9b86ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4936ab50077760dead200ad28e5aab0
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_318ea6ecb9848ae4060f374bee45c8d0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([40, 80], 0, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_15c389d42fdbffb45d180ec284f3e8dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_318ea6ecb9848ae4060f374bee45c8d0
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a558e0793c31ff91eacbba387bdfe25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95109bd6b94b41007c0e1f63000f70f3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_3926a2dc52ea7242a289193e0a6264fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e0c50876d98164601ccb8a967b37066
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7728bdc486306943077ea31df1cefcfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79cd1c1d54ebd9f9c9bdac524eaa7920
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_6a8b89d6ba67997a4f45599dfb6487fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91744f754c892adcead2511f1f945179
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_b7367dc0b3567d5aea3d0bcc502da49e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 1.09589, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7155bf13a3ce1fe2c91658b39fcbe654(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7367dc0b3567d5aea3d0bcc502da49e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_c42ea924c62569f1674b9193c5a808bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15006aad1c5611252c9215622d98664a
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_a6a109a9ba9d7edc0665363334aabe5d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([3800, 81], 0, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5d1768f5bc6df2190f56544fa80106fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6a109a9ba9d7edc0665363334aabe5d
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_3926a2dc52ea7242a289193e0a6264fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e0c50876d98164601ccb8a967b37066
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_cdc6d042a8fba2640bb65cc7017d7074(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 81, paddle.int64, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_495b6c56be1097e9bfb0dde7cea81c8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdc6d042a8fba2640bb65cc7017d7074
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_8ed82428d8739636d019f63588d48544(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 0.25, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3b10c31786c87ceac187ee5a4090a040(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ed82428d8739636d019f63588d48544
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_59884f606a020622bdd05fec914981cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8aab910f89c1b1f773daf7ae13e0507e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_691e735f0b5f309d6a88dd59b3ae1054(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f0dff07192990077d6fb94aa22aa2f9
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a558e0793c31ff91eacbba387bdfe25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95109bd6b94b41007c0e1f63000f70f3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_f633efdab7ab28ab6c0765b90d0bc2de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b77676a4bf5b87dd75109a82f0f76c1
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8c8afa339da46c3c4e981fe750f61605(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a4f96fdfa86d19cdd1c558cd6713c0e
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_f9ba643916772e8424bc886e7c12e0c1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 17, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f14f66574127e13cb86738f88e6475a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9ba643916772e8424bc886e7c12e0c1
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_1a8204a39f0d0f1149f9bfcde90cd701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f684bc3beb79b493a511c02af321449e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_19ebdd27d8f55d774602a71fb3a20612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76ace68dab376620e9fb04790a7cf46b
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_ecabc1f9dced2ff2145690045cbf6c8f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 24, paddle.int32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4e8e2ca8d37136a913360d7aa114cdfd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ecabc1f9dced2ff2145690045cbf6c8f
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_ab6e3f824e07eaa14216b70889b9b4c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59d428e94533601bc16671b552584b86
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_776dea60bffc5234342c42cfbf556a65(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 48, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d7d1341dd97f4439fd14ad93bd403c8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_776dea60bffc5234342c42cfbf556a65
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a591d97808b801aa8b83999c48d59b7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b9bd69b9f40c9f943187eb6d703f689
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_c682d387b92351f95aecae1088d034a2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 96, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_df16027f4ff5afa84be01b9b472adf08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c682d387b92351f95aecae1088d034a2
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_0168b1e8a6ae8a04d43625ae2f3d2d52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7986a3d6d389630a91a1cc7dbb92b680
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_f72d30558ae250318b9d28138bb35f74(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 38, paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5c2d9f66e6dff8469712f40beaaca7ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f72d30558ae250318b9d28138bb35f74
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_cada470e4cbab7e1218beff7167196ad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 25, paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_295950e8cc213d2872aa532c19b84827(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cada470e4cbab7e1218beff7167196ad
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_95cbd657ccc07eabfec381ae4c44f778(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 25, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_421df1aac096369d29b9eea5ed2348a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95cbd657ccc07eabfec381ae4c44f778
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_ef82f8eb3223adc0ae4988cc1558cb00(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 38, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5d13e85f2268dc546e110334f4f552d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef82f8eb3223adc0ae4988cc1558cb00
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_ace018338dc37152bde11588ad34efcb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 2, paddle.int32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2a751c4dbcd2a3004a15a6b8850010e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ace018338dc37152bde11588ad34efcb
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a591d97808b801aa8b83999c48d59b7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b9bd69b9f40c9f943187eb6d703f689
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_3e1765331d2cd655bd1b3b9efcad13b3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 12, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9e26144c82841a0b7024534df87d9470(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e1765331d2cd655bd1b3b9efcad13b3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_ab6e3f824e07eaa14216b70889b9b4c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59d428e94533601bc16671b552584b86
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_4498732842f2d48432721196a11bb7a7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 192, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_73a3053012d555ab7565968b01343b91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4498732842f2d48432721196a11bb7a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_00b5fe9b425704b6a267f9fa46c51e6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ae06a6a28edd1d4ef7daa2193955670
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_f633efdab7ab28ab6c0765b90d0bc2de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b77676a4bf5b87dd75109a82f0f76c1
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8c8afa339da46c3c4e981fe750f61605(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a4f96fdfa86d19cdd1c558cd6713c0e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_f14f66574127e13cb86738f88e6475a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9ba643916772e8424bc886e7c12e0c1
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_1a8204a39f0d0f1149f9bfcde90cd701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f684bc3beb79b493a511c02af321449e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_19ebdd27d8f55d774602a71fb3a20612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76ace68dab376620e9fb04790a7cf46b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a87152b3cd089e715310903b248e3d0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e7f8c962c4c4f3f50acb5310cf1334e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_59884f606a020622bdd05fec914981cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8aab910f89c1b1f773daf7ae13e0507e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_691e735f0b5f309d6a88dd59b3ae1054(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f0dff07192990077d6fb94aa22aa2f9
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_6a8b89d6ba67997a4f45599dfb6487fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91744f754c892adcead2511f1f945179
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a558e0793c31ff91eacbba387bdfe25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95109bd6b94b41007c0e1f63000f70f3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a0ac03445350f26835d65b207edb70ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f44c6866020e1c0100d135816a48333c
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a558e0793c31ff91eacbba387bdfe25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95109bd6b94b41007c0e1f63000f70f3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_f633efdab7ab28ab6c0765b90d0bc2de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b77676a4bf5b87dd75109a82f0f76c1
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_c7b9494b29d2992c39fdd2190f2bc4ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e006158c8e80926f5ba717c1a521302
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_9e26144c82841a0b7024534df87d9470(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e1765331d2cd655bd1b3b9efcad13b3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a87152b3cd089e715310903b248e3d0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e7f8c962c4c4f3f50acb5310cf1334e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2c2593aa06a2163b3136a90856e2581c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2206545d093007dffd5065cf3c441627
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8bb29e2528dbddef14fa8be190b7ec6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8aab409109d28df903667e0731215e8b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_691e735f0b5f309d6a88dd59b3ae1054(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f0dff07192990077d6fb94aa22aa2f9
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_f5ff23ae49d01c04937f0b0289281123(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 80, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4c19113ec85ea8b5778283b5c7d024ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5ff23ae49d01c04937f0b0289281123
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_05652cd19cc06edb45e94fe36db0eff5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], -1, paddle.int32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_52622c67cc4b869e3a000dffd14fa806(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05652cd19cc06edb45e94fe36db0eff5
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_e00ec737e48b8f3fbed7aebce240972f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 14, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_68da5a667b1ad8fb056d46f1531b02ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e00ec737e48b8f3fbed7aebce240972f
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_55ea6b7c692f5ba212f0dcddd11c352a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 384, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_65c3f176920416ee5fc0f9ce712b62c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55ea6b7c692f5ba212f0dcddd11c352a
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_0f3bf925c4d37ec56d06d4cc46ff1577(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 32, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b976a44c2afed26fd64342d44292d9ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f3bf925c4d37ec56d06d4cc46ff1577
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_6a8b89d6ba67997a4f45599dfb6487fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91744f754c892adcead2511f1f945179
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_2b925c6e43960a3192ce46434576ea43(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 96, paddle.int32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fcc1bdec9ee6629b4508caa61b075441(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b925c6e43960a3192ce46434576ea43
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_df16027f4ff5afa84be01b9b472adf08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c682d387b92351f95aecae1088d034a2
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_d7d1341dd97f4439fd14ad93bd403c8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_776dea60bffc5234342c42cfbf556a65
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_b082ff195d3c6b78320cc9ad3e1a27b6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 28, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dd50d2e470c0219f783c065fdbac8978(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b082ff195d3c6b78320cc9ad3e1a27b6
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_73a3053012d555ab7565968b01343b91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4498732842f2d48432721196a11bb7a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_00b5fe9b425704b6a267f9fa46c51e6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ae06a6a28edd1d4ef7daa2193955670
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2c2593aa06a2163b3136a90856e2581c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2206545d093007dffd5065cf3c441627
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a87152b3cd089e715310903b248e3d0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e7f8c962c4c4f3f50acb5310cf1334e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_59884f606a020622bdd05fec914981cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8aab910f89c1b1f773daf7ae13e0507e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_691e735f0b5f309d6a88dd59b3ae1054(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f0dff07192990077d6fb94aa22aa2f9
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_1a8204a39f0d0f1149f9bfcde90cd701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f684bc3beb79b493a511c02af321449e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_19ebdd27d8f55d774602a71fb3a20612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76ace68dab376620e9fb04790a7cf46b
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_b2d96170fc2f7a10d3377262b23f8e10(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 15.99, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_983e4cb517db58e614f66b50f7015f46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2d96170fc2f7a10d3377262b23f8e10
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_f633efdab7ab28ab6c0765b90d0bc2de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b77676a4bf5b87dd75109a82f0f76c1
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8c8afa339da46c3c4e981fe750f61605(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a4f96fdfa86d19cdd1c558cd6713c0e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_f14f66574127e13cb86738f88e6475a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9ba643916772e8424bc886e7c12e0c1
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_1a8204a39f0d0f1149f9bfcde90cd701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f684bc3beb79b493a511c02af321449e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_19ebdd27d8f55d774602a71fb3a20612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76ace68dab376620e9fb04790a7cf46b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_ddde7cefa542fb6dd073cc665e63e2d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df95f02ddb4bb3e8a56884440e59e08f
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_7442a6d34d938334b707351cbecd885a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 30, paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7e1025a4d25b7169bbfb324e980787c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7442a6d34d938334b707351cbecd885a
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_6648c615edc4c4f6ad9ea7efc8cf0a0c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 20, paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0489b99bd19edf1d7d0448ff4357f20f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6648c615edc4c4f6ad9ea7efc8cf0a0c
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b578e6231c9ab10277dee9851a40ac93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d22c0f2354835635eaf0fa0fd6c227f
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_51fb69e6ebb17bb20af6c87b9b25f643(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 30, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_346d126f7540ef168d4edbf153aecfb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51fb69e6ebb17bb20af6c87b9b25f643
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_b876a8e507a388fbf905fb293b76c7df(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 1.05263, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_baa835e5a468aacb3c3bb9f42a13ebf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b876a8e507a388fbf905fb293b76c7df
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_691e735f0b5f309d6a88dd59b3ae1054(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f0dff07192990077d6fb94aa22aa2f9
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_509ed8321457ade6de3d772f1fe18502(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5cf049d907e5d6670f2a060e1085f8d1
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_52622c67cc4b869e3a000dffd14fa806(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05652cd19cc06edb45e94fe36db0eff5
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a558e0793c31ff91eacbba387bdfe25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95109bd6b94b41007c0e1f63000f70f3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_572c650ff72ba7e0d118751410bf2ca9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8fd2faeade559d77a96aa489df0cf0c
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_dc236b766a8ba248dbfd88878591bc92(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 56, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b808c69275a521c71244434f4c934cc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc236b766a8ba248dbfd88878591bc92
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_df16027f4ff5afa84be01b9b472adf08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c682d387b92351f95aecae1088d034a2
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a558e0793c31ff91eacbba387bdfe25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95109bd6b94b41007c0e1f63000f70f3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_6a8b89d6ba67997a4f45599dfb6487fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91744f754c892adcead2511f1f945179
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_d46b754e8d80415852fa55350761b1fe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 64, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_20df9b38cfac761e8fe7ef6b4f2dc78b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d46b754e8d80415852fa55350761b1fe
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2309a2f8454e3f25ed99cebb50f7b2d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_364e0a5000b30cbe7447f3a7b48d65aa
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_f627719d7819b6ec1216dc1ad3105e18(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([4096, 1], 8, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6498061130ae51046ce9564731b15e00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f627719d7819b6ec1216dc1ad3105e18
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_68a825a78a12c2b31f7340d146873488(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4143be70b6a83709b075d6203f2a5577
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_553d5f29dc5959ffeb18466b22d6ff2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e5be3a4d141b05ca8e3c269ba9e2133
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_e4e3ceca9c04bc7114a691a71ab16650(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1024, 1], 16, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_70f97b638c813c6e15fec54b8179c1b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4e3ceca9c04bc7114a691a71ab16650
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_ba8f43bdb0161ad1ffbf6ee3816f0bc9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([256, 1], 32, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f9822f03646c8978763100ce8a1289c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba8f43bdb0161ad1ffbf6ee3816f0bc9
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_1a8204a39f0d0f1149f9bfcde90cd701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f684bc3beb79b493a511c02af321449e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_0168b1e8a6ae8a04d43625ae2f3d2d52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7986a3d6d389630a91a1cc7dbb92b680
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_19ebdd27d8f55d774602a71fb3a20612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76ace68dab376620e9fb04790a7cf46b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_691e735f0b5f309d6a88dd59b3ae1054(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f0dff07192990077d6fb94aa22aa2f9
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_4c19113ec85ea8b5778283b5c7d024ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5ff23ae49d01c04937f0b0289281123
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_52622c67cc4b869e3a000dffd14fa806(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05652cd19cc06edb45e94fe36db0eff5
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a558e0793c31ff91eacbba387bdfe25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95109bd6b94b41007c0e1f63000f70f3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_572c650ff72ba7e0d118751410bf2ca9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8fd2faeade559d77a96aa489df0cf0c
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_543a53c9fa5b260a7b57d663fbf762e4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 1.02564, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8104774f849126c9da9edb125be723fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_543a53c9fa5b260a7b57d663fbf762e4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a87152b3cd089e715310903b248e3d0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e7f8c962c4c4f3f50acb5310cf1334e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_c7b9494b29d2992c39fdd2190f2bc4ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e006158c8e80926f5ba717c1a521302
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_f633efdab7ab28ab6c0765b90d0bc2de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b77676a4bf5b87dd75109a82f0f76c1
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8c8afa339da46c3c4e981fe750f61605(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a4f96fdfa86d19cdd1c558cd6713c0e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_f14f66574127e13cb86738f88e6475a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9ba643916772e8424bc886e7c12e0c1
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_1a8204a39f0d0f1149f9bfcde90cd701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f684bc3beb79b493a511c02af321449e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_19ebdd27d8f55d774602a71fb3a20612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76ace68dab376620e9fb04790a7cf46b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a558e0793c31ff91eacbba387bdfe25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95109bd6b94b41007c0e1f63000f70f3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_f633efdab7ab28ab6c0765b90d0bc2de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b77676a4bf5b87dd75109a82f0f76c1
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_77539020194a2face2ca50729e273684(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 1025, paddle.int32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f233f3e5e94ea5aa2e748c417bc5c6cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77539020194a2face2ca50729e273684
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_6a8b89d6ba67997a4f45599dfb6487fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91744f754c892adcead2511f1f945179
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_c6b549ddd5248dc9054b10d8c5ddfa72(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 1.0101, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_909e1a94cae42717edb5566d93754b54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6b549ddd5248dc9054b10d8c5ddfa72
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_9d875785e12e650ae2c0ae018345ea43(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 0.01, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_004f5c99e6f56d83a66154bfb12d22b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d875785e12e650ae2c0ae018345ea43
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_59884f606a020622bdd05fec914981cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8aab910f89c1b1f773daf7ae13e0507e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_691e735f0b5f309d6a88dd59b3ae1054(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f0dff07192990077d6fb94aa22aa2f9
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_9f30156723f1493e28f82f7bcb842a6f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 2, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e84fd08983ea61db5f6e9d26aecd984c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f30156723f1493e28f82f7bcb842a6f
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_c42ea924c62569f1674b9193c5a808bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15006aad1c5611252c9215622d98664a
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_53b8baee5b9096cabd7e07e4d2134516(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 0.0909091, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_961014caa13d107f94234ae3393105e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53b8baee5b9096cabd7e07e4d2134516
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_c14218e5e4c219495d91681290328607(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2fced7ad1d49ed24bed710a63e84ce2
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_06354aeab0d48eb7c15d22c6b59949ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ae90ea8689aa410ca7d2826b0120d18
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_0168b1e8a6ae8a04d43625ae2f3d2d52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7986a3d6d389630a91a1cc7dbb92b680
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_d40c3580e303579fd03ed34bd13e10ec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 128, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b59595607c11a00c1b97164dd2c3ac97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d40c3580e303579fd03ed34bd13e10ec
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2c2593aa06a2163b3136a90856e2581c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2206545d093007dffd5065cf3c441627
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_1a8204a39f0d0f1149f9bfcde90cd701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f684bc3beb79b493a511c02af321449e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_19ebdd27d8f55d774602a71fb3a20612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76ace68dab376620e9fb04790a7cf46b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_983e4cb517db58e614f66b50f7015f46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2d96170fc2f7a10d3377262b23f8e10
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_5325d0b911c185f407d23d635ff07a4c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 1.11111, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aeb6f161a8745db9fc1370a92ba6abb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5325d0b911c185f407d23d635ff07a4c
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_db7cd1711a66290c002619165e161e7c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 0.1, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_427ddceb6f78e3ee9dd6e4ca8cbe147b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db7cd1711a66290c002619165e161e7c
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_0168b1e8a6ae8a04d43625ae2f3d2d52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7986a3d6d389630a91a1cc7dbb92b680
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_039fc396b102472fb4a6e5a0fcecc254(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_993f0e791c7377a5b0e132b3805d13e2
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_0168b1e8a6ae8a04d43625ae2f3d2d52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7986a3d6d389630a91a1cc7dbb92b680
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_358e397f6796c1e46abb7841d5e5f813(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 1.19403, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d078ab9abcef75977944c12414fc36fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_358e397f6796c1e46abb7841d5e5f813
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_dcba987f158dadb212b35397b2cf1754(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1, 256], 0, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d25f64d991d9abe5a5f5552769b59b71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dcba987f158dadb212b35397b2cf1754
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_4197f079bfdabda80d50107bd91f9e58(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1, 501, 30], 0, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a05e5c132e6fed8e81e2cc536e0869a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4197f079bfdabda80d50107bd91f9e58
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_8f84b7b417d08da8c2b987d6448a2c7c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1, 501, 4], 0, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_623a5c28f6f5f9097689418633af0b7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f84b7b417d08da8c2b987d6448a2c7c
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_039fc396b102472fb4a6e5a0fcecc254(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_993f0e791c7377a5b0e132b3805d13e2
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2a751c4dbcd2a3004a15a6b8850010e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ace018338dc37152bde11588ad34efcb
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a591d97808b801aa8b83999c48d59b7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b9bd69b9f40c9f943187eb6d703f689
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_ab6e3f824e07eaa14216b70889b9b4c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59d428e94533601bc16671b552584b86
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_9e26144c82841a0b7024534df87d9470(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e1765331d2cd655bd1b3b9efcad13b3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_73a3053012d555ab7565968b01343b91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4498732842f2d48432721196a11bb7a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2a751c4dbcd2a3004a15a6b8850010e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ace018338dc37152bde11588ad34efcb
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a558e0793c31ff91eacbba387bdfe25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95109bd6b94b41007c0e1f63000f70f3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_572c650ff72ba7e0d118751410bf2ca9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8fd2faeade559d77a96aa489df0cf0c
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_3b10c31786c87ceac187ee5a4090a040(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ed82428d8739636d019f63588d48544
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_0168b1e8a6ae8a04d43625ae2f3d2d52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7986a3d6d389630a91a1cc7dbb92b680
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_aeb6f161a8745db9fc1370a92ba6abb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5325d0b911c185f407d23d635ff07a4c
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_427ddceb6f78e3ee9dd6e4ca8cbe147b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db7cd1711a66290c002619165e161e7c
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_aabcd7dcd2477b681b01226445ecb077(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64194509319894b799f6df30002a5cb
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8bb29e2528dbddef14fa8be190b7ec6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8aab409109d28df903667e0731215e8b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_dd58cf7f7138d35e576a7c1a3ebcde1f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([15200, 81], 0, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_00978454ad20888531591f23623d4bc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd58cf7f7138d35e576a7c1a3ebcde1f
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_3926a2dc52ea7242a289193e0a6264fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e0c50876d98164601ccb8a967b37066
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_495b6c56be1097e9bfb0dde7cea81c8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdc6d042a8fba2640bb65cc7017d7074
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_2cf677087dd01579c4b6a4890caad9a7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 1280, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4e5b0fe8e55b1ce0048ec7b498a4b763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cf677087dd01579c4b6a4890caad9a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_e9ea5fcf8655f8d4e9f7072972f3a3f7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([15200, 80], 0, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4ebce662b0366c99c1989efdbbb04a19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9ea5fcf8655f8d4e9f7072972f3a3f7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a558e0793c31ff91eacbba387bdfe25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95109bd6b94b41007c0e1f63000f70f3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_3926a2dc52ea7242a289193e0a6264fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e0c50876d98164601ccb8a967b37066
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7728bdc486306943077ea31df1cefcfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79cd1c1d54ebd9f9c9bdac524eaa7920
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_fcc1bdec9ee6629b4508caa61b075441(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b925c6e43960a3192ce46434576ea43
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_df16027f4ff5afa84be01b9b472adf08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c682d387b92351f95aecae1088d034a2
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_d7d1341dd97f4439fd14ad93bd403c8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_776dea60bffc5234342c42cfbf556a65
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_0168b1e8a6ae8a04d43625ae2f3d2d52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7986a3d6d389630a91a1cc7dbb92b680
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_e13f875b3cb7da10ff338fc779ce8158(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8b26ad00a514dfeff3b7c530feb5e6e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b0a5f722ee7e58afbe2e2f5394154f9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4c31049513dc61e0ef38bf74bcd91b4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_414cbe2c49ddbc62cc270018556d3439(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_900902bf4b6bf411697f3e4135095bdc
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_0168b1e8a6ae8a04d43625ae2f3d2d52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7986a3d6d389630a91a1cc7dbb92b680
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_509ed8321457ade6de3d772f1fe18502(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5cf049d907e5d6670f2a060e1085f8d1
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a558e0793c31ff91eacbba387bdfe25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95109bd6b94b41007c0e1f63000f70f3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_572c650ff72ba7e0d118751410bf2ca9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8fd2faeade559d77a96aa489df0cf0c
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_52622c67cc4b869e3a000dffd14fa806(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05652cd19cc06edb45e94fe36db0eff5
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_9e26144c82841a0b7024534df87d9470(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e1765331d2cd655bd1b3b9efcad13b3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8bb29e2528dbddef14fa8be190b7ec6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8aab409109d28df903667e0731215e8b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_4e8e2ca8d37136a913360d7aa114cdfd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ecabc1f9dced2ff2145690045cbf6c8f
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_ab6e3f824e07eaa14216b70889b9b4c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59d428e94533601bc16671b552584b86
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_d7d1341dd97f4439fd14ad93bd403c8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_776dea60bffc5234342c42cfbf556a65
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a591d97808b801aa8b83999c48d59b7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b9bd69b9f40c9f943187eb6d703f689
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_df16027f4ff5afa84be01b9b472adf08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c682d387b92351f95aecae1088d034a2
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2a751c4dbcd2a3004a15a6b8850010e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ace018338dc37152bde11588ad34efcb
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a591d97808b801aa8b83999c48d59b7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b9bd69b9f40c9f943187eb6d703f689
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_9e26144c82841a0b7024534df87d9470(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e1765331d2cd655bd1b3b9efcad13b3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_ab6e3f824e07eaa14216b70889b9b4c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59d428e94533601bc16671b552584b86
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_73a3053012d555ab7565968b01343b91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4498732842f2d48432721196a11bb7a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_4d7b7244bc884b86649fb51f2a9b86ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4936ab50077760dead200ad28e5aab0
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b976a44c2afed26fd64342d44292d9ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f3bf925c4d37ec56d06d4cc46ff1577
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_f633efdab7ab28ab6c0765b90d0bc2de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b77676a4bf5b87dd75109a82f0f76c1
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a591d97808b801aa8b83999c48d59b7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b9bd69b9f40c9f943187eb6d703f689
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_c42ea924c62569f1674b9193c5a808bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15006aad1c5611252c9215622d98664a
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_e13f875b3cb7da10ff338fc779ce8158(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8b26ad00a514dfeff3b7c530feb5e6e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_485dde1c74b3455de459aa6fbb8feed3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9d0e23085c4d0c96e3e881d0902c120
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_909e1a94cae42717edb5566d93754b54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6b549ddd5248dc9054b10d8c5ddfa72
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_004f5c99e6f56d83a66154bfb12d22b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d875785e12e650ae2c0ae018345ea43
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_0168b1e8a6ae8a04d43625ae2f3d2d52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7986a3d6d389630a91a1cc7dbb92b680
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_9e26144c82841a0b7024534df87d9470(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e1765331d2cd655bd1b3b9efcad13b3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8bb29e2528dbddef14fa8be190b7ec6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8aab409109d28df903667e0731215e8b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_fcc1bdec9ee6629b4508caa61b075441(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b925c6e43960a3192ce46434576ea43
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_df16027f4ff5afa84be01b9b472adf08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c682d387b92351f95aecae1088d034a2
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_d7d1341dd97f4439fd14ad93bd403c8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_776dea60bffc5234342c42cfbf556a65
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_0168b1e8a6ae8a04d43625ae2f3d2d52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7986a3d6d389630a91a1cc7dbb92b680
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_912c945ef1399103a3cf92322228393d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 0.405285, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e96d669c31d4e1d0ec727e1fd956ae5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_912c945ef1399103a3cf92322228393d
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_556f1f58ed5d61f407f0e1e1541b6b4f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 10, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8982c65bf1c175616519ac61228e5cb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_556f1f58ed5d61f407f0e1e1541b6b4f
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_c42ea924c62569f1674b9193c5a808bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15006aad1c5611252c9215622d98664a
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_1a8204a39f0d0f1149f9bfcde90cd701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f684bc3beb79b493a511c02af321449e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_19ebdd27d8f55d774602a71fb3a20612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76ace68dab376620e9fb04790a7cf46b
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_01bae6e214c1a0ec37c124f2cae69160(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], -2, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d98825011d053417561fc777c57f7e03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01bae6e214c1a0ec37c124f2cae69160
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_983e4cb517db58e614f66b50f7015f46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2d96170fc2f7a10d3377262b23f8e10
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_aeb6f161a8745db9fc1370a92ba6abb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5325d0b911c185f407d23d635ff07a4c
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_427ddceb6f78e3ee9dd6e4ca8cbe147b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db7cd1711a66290c002619165e161e7c
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_aeb6f161a8745db9fc1370a92ba6abb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5325d0b911c185f407d23d635ff07a4c
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_427ddceb6f78e3ee9dd6e4ca8cbe147b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db7cd1711a66290c002619165e161e7c
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_fcc1bdec9ee6629b4508caa61b075441(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b925c6e43960a3192ce46434576ea43
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_df16027f4ff5afa84be01b9b472adf08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c682d387b92351f95aecae1088d034a2
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_d7d1341dd97f4439fd14ad93bd403c8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_776dea60bffc5234342c42cfbf556a65
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_1a8204a39f0d0f1149f9bfcde90cd701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f684bc3beb79b493a511c02af321449e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_9c0aca59885d2ef4005bfa5c6591331f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 0, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_96e40d5b754a92c01327af4b7ef9006e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c0aca59885d2ef4005bfa5c6591331f
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a591d97808b801aa8b83999c48d59b7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b9bd69b9f40c9f943187eb6d703f689
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_25b952fe4398a4e09ea4da83f66a99f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_606892bbebc398b07aa0b00da1b29a42
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_4d80f979a187595b4825dbd2b7aa8d30(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 72, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0ec9b9857f332224bc24112af3ccf792(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d80f979a187595b4825dbd2b7aa8d30
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_e84fd08983ea61db5f6e9d26aecd984c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f30156723f1493e28f82f7bcb842a6f
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_039fc396b102472fb4a6e5a0fcecc254(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_993f0e791c7377a5b0e132b3805d13e2
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_485dde1c74b3455de459aa6fbb8feed3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9d0e23085c4d0c96e3e881d0902c120
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_f633efdab7ab28ab6c0765b90d0bc2de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b77676a4bf5b87dd75109a82f0f76c1
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8c8afa339da46c3c4e981fe750f61605(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a4f96fdfa86d19cdd1c558cd6713c0e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_f14f66574127e13cb86738f88e6475a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9ba643916772e8424bc886e7c12e0c1
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_1a8204a39f0d0f1149f9bfcde90cd701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f684bc3beb79b493a511c02af321449e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_19ebdd27d8f55d774602a71fb3a20612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76ace68dab376620e9fb04790a7cf46b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_6a8b89d6ba67997a4f45599dfb6487fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91744f754c892adcead2511f1f945179
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_a941d081b4fe94e3cbad79efd4f2f9a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 0.05, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cfa6f473ec7312afab0b33bdb2f0e438(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a941d081b4fe94e3cbad79efd4f2f9a6
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_aeb6f161a8745db9fc1370a92ba6abb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5325d0b911c185f407d23d635ff07a4c
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_427ddceb6f78e3ee9dd6e4ca8cbe147b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db7cd1711a66290c002619165e161e7c
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_52622c67cc4b869e3a000dffd14fa806(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05652cd19cc06edb45e94fe36db0eff5
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_4c19113ec85ea8b5778283b5c7d024ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5ff23ae49d01c04937f0b0289281123
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2309a2f8454e3f25ed99cebb50f7b2d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_364e0a5000b30cbe7447f3a7b48d65aa
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_7442c76e3e9cbd8a2bb11ed23b0ab186(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([6400, 1], 8, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dd0c8a984c7f25a506019c501f25abbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7442c76e3e9cbd8a2bb11ed23b0ab186
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_7ead18ba79f15b64b3f85bb59436d124(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 40, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_97bade83316c94f2302bfd17953b5cb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ead18ba79f15b64b3f85bb59436d124
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_553d5f29dc5959ffeb18466b22d6ff2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e5be3a4d141b05ca8e3c269ba9e2133
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_f6e8ef4be72764bff0f0dd45b0673f4b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1600, 1], 16, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cad37feed426f8fe9f771aff5d2b0973(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6e8ef4be72764bff0f0dd45b0673f4b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_e674bd5969b49291fe1545dcad38919e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96190683c5a4edff8d8977d736e15e7d
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_68a825a78a12c2b31f7340d146873488(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4143be70b6a83709b075d6203f2a5577
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_97d88b6f6d976871d697ac48b986a38a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([400, 1], 32, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7ab85f56cc537492868a58842303de80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97d88b6f6d976871d697ac48b986a38a
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_1a8204a39f0d0f1149f9bfcde90cd701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f684bc3beb79b493a511c02af321449e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_0168b1e8a6ae8a04d43625ae2f3d2d52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7986a3d6d389630a91a1cc7dbb92b680
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_19ebdd27d8f55d774602a71fb3a20612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76ace68dab376620e9fb04790a7cf46b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_6883b47cae0f46a2172898580cbe248d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 3, paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d30c38ce888186114da428dbca66a2be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6883b47cae0f46a2172898580cbe248d
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_c7b9494b29d2992c39fdd2190f2bc4ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e006158c8e80926f5ba717c1a521302
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_cc15241898e44f389669e2a2824aa864(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 1e-10, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b9d5d2856a40f417a0de26ea9d2ad175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc15241898e44f389669e2a2824aa864
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b808c69275a521c71244434f4c934cc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc236b766a8ba248dbfd88878591bc92
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_df16027f4ff5afa84be01b9b472adf08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c682d387b92351f95aecae1088d034a2
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_485dde1c74b3455de459aa6fbb8feed3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9d0e23085c4d0c96e3e881d0902c120
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_3b10c31786c87ceac187ee5a4090a040(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ed82428d8739636d019f63588d48544
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_59884f606a020622bdd05fec914981cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8aab910f89c1b1f773daf7ae13e0507e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_691e735f0b5f309d6a88dd59b3ae1054(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f0dff07192990077d6fb94aa22aa2f9
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_827d102786d013b40bdd26b7ef63298e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a5da53036182f346c07ea5fc3ed8632
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_4e8fa2e227d494a3f6a8522b5905f72f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7209f99ccd37b7e49637ebaa32a52f80
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_68a825a78a12c2b31f7340d146873488(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4143be70b6a83709b075d6203f2a5577
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_553d5f29dc5959ffeb18466b22d6ff2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e5be3a4d141b05ca8e3c269ba9e2133
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2309a2f8454e3f25ed99cebb50f7b2d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_364e0a5000b30cbe7447f3a7b48d65aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_6a8b89d6ba67997a4f45599dfb6487fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91744f754c892adcead2511f1f945179
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a558e0793c31ff91eacbba387bdfe25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95109bd6b94b41007c0e1f63000f70f3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_f633efdab7ab28ab6c0765b90d0bc2de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b77676a4bf5b87dd75109a82f0f76c1
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8c8afa339da46c3c4e981fe750f61605(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a4f96fdfa86d19cdd1c558cd6713c0e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_f14f66574127e13cb86738f88e6475a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9ba643916772e8424bc886e7c12e0c1
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_1a8204a39f0d0f1149f9bfcde90cd701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f684bc3beb79b493a511c02af321449e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_19ebdd27d8f55d774602a71fb3a20612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76ace68dab376620e9fb04790a7cf46b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_1a8204a39f0d0f1149f9bfcde90cd701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f684bc3beb79b493a511c02af321449e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_19ebdd27d8f55d774602a71fb3a20612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76ace68dab376620e9fb04790a7cf46b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_983e4cb517db58e614f66b50f7015f46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2d96170fc2f7a10d3377262b23f8e10
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_375969ca7497e3f3e0a6e699730622b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56e00f90b1ee2c2ad63326a5d6bb69ed
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_e84fd08983ea61db5f6e9d26aecd984c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f30156723f1493e28f82f7bcb842a6f
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2e624ca7b64813c0ca2b1d5c90749ad6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3447c8d1bdff45772f6c683747ac318b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_4c19113ec85ea8b5778283b5c7d024ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5ff23ae49d01c04937f0b0289281123
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_23c6728e486db9868179309e07031c18(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 81, paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ba3cb5e743d2e7f0b670806f29c9c856(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23c6728e486db9868179309e07031c18
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_dd50d2e470c0219f783c065fdbac8978(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b082ff195d3c6b78320cc9ad3e1a27b6
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_73a3053012d555ab7565968b01343b91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4498732842f2d48432721196a11bb7a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_6a8b89d6ba67997a4f45599dfb6487fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91744f754c892adcead2511f1f945179
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_96e40d5b754a92c01327af4b7ef9006e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c0aca59885d2ef4005bfa5c6591331f
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_4d7b7244bc884b86649fb51f2a9b86ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4936ab50077760dead200ad28e5aab0
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_f633efdab7ab28ab6c0765b90d0bc2de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b77676a4bf5b87dd75109a82f0f76c1
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_0386ad510cd9c3c2af491cc475af7f34(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 512, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b77bc9b2933d557d58d5888c1957ea70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0386ad510cd9c3c2af491cc475af7f34
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_485dde1c74b3455de459aa6fbb8feed3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9d0e23085c4d0c96e3e881d0902c120
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_3b10c31786c87ceac187ee5a4090a040(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ed82428d8739636d019f63588d48544
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_ca4ecbfb76554c3b640e22788c87fdae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 4, paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ab335bb7ca75cde65b87c6f234a71c68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca4ecbfb76554c3b640e22788c87fdae
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_1a8204a39f0d0f1149f9bfcde90cd701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f684bc3beb79b493a511c02af321449e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2a751c4dbcd2a3004a15a6b8850010e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ace018338dc37152bde11588ad34efcb
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_be193d9df3b0d1f835bdd91b2e05a794(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([2], 0, paddle.int32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_47553401b84b37b0889185bdcdfa3928(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be193d9df3b0d1f835bdd91b2e05a794
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_96e40d5b754a92c01327af4b7ef9006e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c0aca59885d2ef4005bfa5c6591331f
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_4d7b7244bc884b86649fb51f2a9b86ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4936ab50077760dead200ad28e5aab0
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_f633efdab7ab28ab6c0765b90d0bc2de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b77676a4bf5b87dd75109a82f0f76c1
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b77bc9b2933d557d58d5888c1957ea70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0386ad510cd9c3c2af491cc475af7f34
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_6a8b89d6ba67997a4f45599dfb6487fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91744f754c892adcead2511f1f945179
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a558e0793c31ff91eacbba387bdfe25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95109bd6b94b41007c0e1f63000f70f3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_572c650ff72ba7e0d118751410bf2ca9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8fd2faeade559d77a96aa489df0cf0c
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a558e0793c31ff91eacbba387bdfe25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95109bd6b94b41007c0e1f63000f70f3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_572c650ff72ba7e0d118751410bf2ca9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8fd2faeade559d77a96aa489df0cf0c
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_26bb36efd7f41cb0c2357ebd21b5593c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([2204, 80], 0, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_70975880e4e7f28a672aa368c4b6e216(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26bb36efd7f41cb0c2357ebd21b5593c
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a558e0793c31ff91eacbba387bdfe25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95109bd6b94b41007c0e1f63000f70f3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_3926a2dc52ea7242a289193e0a6264fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e0c50876d98164601ccb8a967b37066
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7728bdc486306943077ea31df1cefcfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79cd1c1d54ebd9f9c9bdac524eaa7920
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_3b10c31786c87ceac187ee5a4090a040(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ed82428d8739636d019f63588d48544
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_f633efdab7ab28ab6c0765b90d0bc2de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b77676a4bf5b87dd75109a82f0f76c1
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_c7b9494b29d2992c39fdd2190f2bc4ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e006158c8e80926f5ba717c1a521302
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_179832a90b4a3da398a66868e37ff706(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 6, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1566d0da727dc0b417f0a6449bad4860(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_179832a90b4a3da398a66868e37ff706
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a87152b3cd089e715310903b248e3d0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e7f8c962c4c4f3f50acb5310cf1334e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2c2593aa06a2163b3136a90856e2581c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2206545d093007dffd5065cf3c441627
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_65c3f176920416ee5fc0f9ce712b62c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55ea6b7c692f5ba212f0dcddd11c352a
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_59884f606a020622bdd05fec914981cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8aab910f89c1b1f773daf7ae13e0507e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_691e735f0b5f309d6a88dd59b3ae1054(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f0dff07192990077d6fb94aa22aa2f9
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a558e0793c31ff91eacbba387bdfe25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95109bd6b94b41007c0e1f63000f70f3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_572c650ff72ba7e0d118751410bf2ca9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8fd2faeade559d77a96aa489df0cf0c
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_96e40d5b754a92c01327af4b7ef9006e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c0aca59885d2ef4005bfa5c6591331f
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b77bc9b2933d557d58d5888c1957ea70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0386ad510cd9c3c2af491cc475af7f34
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_e84fd08983ea61db5f6e9d26aecd984c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f30156723f1493e28f82f7bcb842a6f
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_d30c38ce888186114da428dbca66a2be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6883b47cae0f46a2172898580cbe248d
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_e84fd08983ea61db5f6e9d26aecd984c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f30156723f1493e28f82f7bcb842a6f
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8104774f849126c9da9edb125be723fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_543a53c9fa5b260a7b57d663fbf762e4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_59884f606a020622bdd05fec914981cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8aab910f89c1b1f773daf7ae13e0507e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_691e735f0b5f309d6a88dd59b3ae1054(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f0dff07192990077d6fb94aa22aa2f9
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_96e40d5b754a92c01327af4b7ef9006e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c0aca59885d2ef4005bfa5c6591331f
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_0306820bb9f4fb1dcda11c092a1535fd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 150, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3f17ebbc2c2865a8c15c95cbe9c2dde9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0306820bb9f4fb1dcda11c092a1535fd
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_3a3e964d77b7f66a20b169835803f0e2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 14, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_861cbf69d21ea92d254697779bfa8bb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a3e964d77b7f66a20b169835803f0e2
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_68a825a78a12c2b31f7340d146873488(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4143be70b6a83709b075d6203f2a5577
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_557eb08a7aed82b3a9e7ab912674c391(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([196, 1], 32, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a43e28094d05ef7df43a35082ac6958d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_557eb08a7aed82b3a9e7ab912674c391
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_0798de2582ee035d551874bb001d9c4e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 28, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8b8842437abbb902ec227ee1132f589d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0798de2582ee035d551874bb001d9c4e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_553d5f29dc5959ffeb18466b22d6ff2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e5be3a4d141b05ca8e3c269ba9e2133
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_14761325d15ebb90ca0110d729801388(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([784, 1], 16, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_02317b7c4e75c6ce50bdd2b2dfcee3ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_14761325d15ebb90ca0110d729801388
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_44d53c6e7d310121f8c24235a6683534(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 56, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1f96103a42346a33ddacc39c454982b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44d53c6e7d310121f8c24235a6683534
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2309a2f8454e3f25ed99cebb50f7b2d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_364e0a5000b30cbe7447f3a7b48d65aa
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_abf5835450168f4ac0904fdd571c5e22(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([3136, 1], 8, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f9f4d028515c8ac2723aa69d36f352d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_abf5835450168f4ac0904fdd571c5e22
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_0168b1e8a6ae8a04d43625ae2f3d2d52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7986a3d6d389630a91a1cc7dbb92b680
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_1a8204a39f0d0f1149f9bfcde90cd701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f684bc3beb79b493a511c02af321449e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_4d7b7244bc884b86649fb51f2a9b86ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4936ab50077760dead200ad28e5aab0
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2309a2f8454e3f25ed99cebb50f7b2d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_364e0a5000b30cbe7447f3a7b48d65aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_96e40d5b754a92c01327af4b7ef9006e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c0aca59885d2ef4005bfa5c6591331f
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b77bc9b2933d557d58d5888c1957ea70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0386ad510cd9c3c2af491cc475af7f34
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_691e735f0b5f309d6a88dd59b3ae1054(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f0dff07192990077d6fb94aa22aa2f9
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_68da5a667b1ad8fb056d46f1531b02ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e00ec737e48b8f3fbed7aebce240972f
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_65c3f176920416ee5fc0f9ce712b62c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55ea6b7c692f5ba212f0dcddd11c352a
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_4d7b7244bc884b86649fb51f2a9b86ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4936ab50077760dead200ad28e5aab0
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2309a2f8454e3f25ed99cebb50f7b2d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_364e0a5000b30cbe7447f3a7b48d65aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_96e40d5b754a92c01327af4b7ef9006e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c0aca59885d2ef4005bfa5c6591331f
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b77bc9b2933d557d58d5888c1957ea70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0386ad510cd9c3c2af491cc475af7f34
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_691e735f0b5f309d6a88dd59b3ae1054(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f0dff07192990077d6fb94aa22aa2f9
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_442577ef77ff258a663a684cfb305277(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([70, 81], 0, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a8661e329d0a173a34a68b6f947052b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_442577ef77ff258a663a684cfb305277
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_3926a2dc52ea7242a289193e0a6264fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e0c50876d98164601ccb8a967b37066
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_495b6c56be1097e9bfb0dde7cea81c8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdc6d042a8fba2640bb65cc7017d7074
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_375969ca7497e3f3e0a6e699730622b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56e00f90b1ee2c2ad63326a5d6bb69ed
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2e624ca7b64813c0ca2b1d5c90749ad6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3447c8d1bdff45772f6c683747ac318b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_e674bd5969b49291fe1545dcad38919e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96190683c5a4edff8d8977d736e15e7d
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_4272819eb3706703baa0eaf500a2fa16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cfe5d6500f99a977d9da41a9a7128a18
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_7ba7b851b59b9c28edbb58197556cfa7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([551, 80], 0, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ec3f9ac6bb95d23ac0288e3008cd0539(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ba7b851b59b9c28edbb58197556cfa7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a558e0793c31ff91eacbba387bdfe25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95109bd6b94b41007c0e1f63000f70f3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_3926a2dc52ea7242a289193e0a6264fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e0c50876d98164601ccb8a967b37066
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7728bdc486306943077ea31df1cefcfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79cd1c1d54ebd9f9c9bdac524eaa7920
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_584d2eb8563b949b4ede75832dba55df(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 160, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_287dfb6cf3bf07919ba50ac5fd420eec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_584d2eb8563b949b4ede75832dba55df
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_c42ea924c62569f1674b9193c5a808bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15006aad1c5611252c9215622d98664a
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_59884f606a020622bdd05fec914981cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8aab910f89c1b1f773daf7ae13e0507e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_691e735f0b5f309d6a88dd59b3ae1054(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f0dff07192990077d6fb94aa22aa2f9
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_0d6f994bafb2682c663ef91b4de6e89f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 0.0015625, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_700b9dd587c9a26c5951ea0b0f61f7fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d6f994bafb2682c663ef91b4de6e89f
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_5283b63ff801decffa2c25efb0eac850(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 1, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_18b82a80807defd17e6ff1ab9468fe72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5283b63ff801decffa2c25efb0eac850
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_4d7b7244bc884b86649fb51f2a9b86ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4936ab50077760dead200ad28e5aab0
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b976a44c2afed26fd64342d44292d9ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f3bf925c4d37ec56d06d4cc46ff1577
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_f633efdab7ab28ab6c0765b90d0bc2de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b77676a4bf5b87dd75109a82f0f76c1
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a591d97808b801aa8b83999c48d59b7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b9bd69b9f40c9f943187eb6d703f689
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_c42ea924c62569f1674b9193c5a808bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15006aad1c5611252c9215622d98664a
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_e13f875b3cb7da10ff338fc779ce8158(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8b26ad00a514dfeff3b7c530feb5e6e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a558e0793c31ff91eacbba387bdfe25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95109bd6b94b41007c0e1f63000f70f3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_572c650ff72ba7e0d118751410bf2ca9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8fd2faeade559d77a96aa489df0cf0c
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b976a44c2afed26fd64342d44292d9ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f3bf925c4d37ec56d06d4cc46ff1577
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_6a8b89d6ba67997a4f45599dfb6487fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91744f754c892adcead2511f1f945179
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_2c6800769c72ccc98a58a9abb187e893(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([247, 80], 0, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9d0090b408cee0d8e9a549c94f5aac8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c6800769c72ccc98a58a9abb187e893
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a558e0793c31ff91eacbba387bdfe25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95109bd6b94b41007c0e1f63000f70f3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_3926a2dc52ea7242a289193e0a6264fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e0c50876d98164601ccb8a967b37066
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7728bdc486306943077ea31df1cefcfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79cd1c1d54ebd9f9c9bdac524eaa7920
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_0168b1e8a6ae8a04d43625ae2f3d2d52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7986a3d6d389630a91a1cc7dbb92b680
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_039fc396b102472fb4a6e5a0fcecc254(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_993f0e791c7377a5b0e132b3805d13e2
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_263babf8c6879a467ace92b7e1193f2d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([950, 80], 0, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e918680b4f22825ccd6cef747d687200(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_263babf8c6879a467ace92b7e1193f2d
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a558e0793c31ff91eacbba387bdfe25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95109bd6b94b41007c0e1f63000f70f3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_3926a2dc52ea7242a289193e0a6264fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e0c50876d98164601ccb8a967b37066
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7728bdc486306943077ea31df1cefcfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79cd1c1d54ebd9f9c9bdac524eaa7920
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a591d97808b801aa8b83999c48d59b7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b9bd69b9f40c9f943187eb6d703f689
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_7b38dbe8018bb1bb11bb624adf0ea1ea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 232, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9606110d8f6450e7744f4dd2137eb824(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b38dbe8018bb1bb11bb624adf0ea1ea
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_35b3b562036bc9f4ac5b3bb1263799c8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 464, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_187b23b9faed9bdd319c79330e073916(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35b3b562036bc9f4ac5b3bb1263799c8
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2c2593aa06a2163b3136a90856e2581c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2206545d093007dffd5065cf3c441627
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a591d97808b801aa8b83999c48d59b7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b9bd69b9f40c9f943187eb6d703f689
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_c2a8e9c332f4606e801870f01e606275(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 16, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_af0a3ba48662d946b8e7d4dd3698771e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2a8e9c332f4606e801870f01e606275
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b976a44c2afed26fd64342d44292d9ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f3bf925c4d37ec56d06d4cc46ff1577
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a558e0793c31ff91eacbba387bdfe25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95109bd6b94b41007c0e1f63000f70f3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a558e0793c31ff91eacbba387bdfe25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95109bd6b94b41007c0e1f63000f70f3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_572c650ff72ba7e0d118751410bf2ca9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8fd2faeade559d77a96aa489df0cf0c
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b976a44c2afed26fd64342d44292d9ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f3bf925c4d37ec56d06d4cc46ff1577
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_c42ea924c62569f1674b9193c5a808bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15006aad1c5611252c9215622d98664a
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_19d0e3cbe63b84c62cee78b7412ba7f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee40ea0e78a7cdebaef17925b3c1562a
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_1a8204a39f0d0f1149f9bfcde90cd701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f684bc3beb79b493a511c02af321449e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_1a8204a39f0d0f1149f9bfcde90cd701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f684bc3beb79b493a511c02af321449e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_19ebdd27d8f55d774602a71fb3a20612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76ace68dab376620e9fb04790a7cf46b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_983e4cb517db58e614f66b50f7015f46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2d96170fc2f7a10d3377262b23f8e10
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_287dfb6cf3bf07919ba50ac5fd420eec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_584d2eb8563b949b4ede75832dba55df
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_68da5a667b1ad8fb056d46f1531b02ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e00ec737e48b8f3fbed7aebce240972f
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_65c3f176920416ee5fc0f9ce712b62c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55ea6b7c692f5ba212f0dcddd11c352a
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2a751c4dbcd2a3004a15a6b8850010e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ace018338dc37152bde11588ad34efcb
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a591d97808b801aa8b83999c48d59b7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b9bd69b9f40c9f943187eb6d703f689
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_ab6e3f824e07eaa14216b70889b9b4c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59d428e94533601bc16671b552584b86
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_9e26144c82841a0b7024534df87d9470(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e1765331d2cd655bd1b3b9efcad13b3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_73a3053012d555ab7565968b01343b91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4498732842f2d48432721196a11bb7a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a558e0793c31ff91eacbba387bdfe25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95109bd6b94b41007c0e1f63000f70f3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_572c650ff72ba7e0d118751410bf2ca9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8fd2faeade559d77a96aa489df0cf0c
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b47a884d7150d786fb53dafcad45680c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c636f5fff1fc99c1cfb715dab7b2c2d2
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_553d5f29dc5959ffeb18466b22d6ff2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e5be3a4d141b05ca8e3c269ba9e2133
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_0168b1e8a6ae8a04d43625ae2f3d2d52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7986a3d6d389630a91a1cc7dbb92b680
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_e59f4238beb4826102c0dc4bfc97068d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([8816, 80], 0, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9346268de8df0e72299a66b3462d55e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e59f4238beb4826102c0dc4bfc97068d
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a558e0793c31ff91eacbba387bdfe25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95109bd6b94b41007c0e1f63000f70f3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_3926a2dc52ea7242a289193e0a6264fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e0c50876d98164601ccb8a967b37066
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7728bdc486306943077ea31df1cefcfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79cd1c1d54ebd9f9c9bdac524eaa7920
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_6a8b89d6ba67997a4f45599dfb6487fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91744f754c892adcead2511f1f945179
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_414cbe2c49ddbc62cc270018556d3439(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_900902bf4b6bf411697f3e4135095bdc
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_0168b1e8a6ae8a04d43625ae2f3d2d52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7986a3d6d389630a91a1cc7dbb92b680
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_509ed8321457ade6de3d772f1fe18502(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5cf049d907e5d6670f2a060e1085f8d1
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_96e40d5b754a92c01327af4b7ef9006e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c0aca59885d2ef4005bfa5c6591331f
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a591d97808b801aa8b83999c48d59b7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b9bd69b9f40c9f943187eb6d703f689
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_0ec9b9857f332224bc24112af3ccf792(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d80f979a187595b4825dbd2b7aa8d30
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_30f503edd74044748158655f6105bf0a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 144, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c59446f04aa0ba344faf0d591d5a048d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30f503edd74044748158655f6105bf0a
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_4d7b7244bc884b86649fb51f2a9b86ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4936ab50077760dead200ad28e5aab0
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_553d5f29dc5959ffeb18466b22d6ff2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e5be3a4d141b05ca8e3c269ba9e2133
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b808c69275a521c71244434f4c934cc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc236b766a8ba248dbfd88878591bc92
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_df16027f4ff5afa84be01b9b472adf08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c682d387b92351f95aecae1088d034a2
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_1a8204a39f0d0f1149f9bfcde90cd701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f684bc3beb79b493a511c02af321449e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_19ebdd27d8f55d774602a71fb3a20612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76ace68dab376620e9fb04790a7cf46b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_983e4cb517db58e614f66b50f7015f46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2d96170fc2f7a10d3377262b23f8e10
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_c42ea924c62569f1674b9193c5a808bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15006aad1c5611252c9215622d98664a
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a0ac03445350f26835d65b207edb70ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f44c6866020e1c0100d135816a48333c
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a558e0793c31ff91eacbba387bdfe25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95109bd6b94b41007c0e1f63000f70f3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7728bdc486306943077ea31df1cefcfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79cd1c1d54ebd9f9c9bdac524eaa7920
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_1a8204a39f0d0f1149f9bfcde90cd701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f684bc3beb79b493a511c02af321449e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_19ebdd27d8f55d774602a71fb3a20612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76ace68dab376620e9fb04790a7cf46b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_983e4cb517db58e614f66b50f7015f46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2d96170fc2f7a10d3377262b23f8e10
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a87152b3cd089e715310903b248e3d0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e7f8c962c4c4f3f50acb5310cf1334e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2c2593aa06a2163b3136a90856e2581c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2206545d093007dffd5065cf3c441627
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_96e40d5b754a92c01327af4b7ef9006e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c0aca59885d2ef4005bfa5c6591331f
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_3f17ebbc2c2865a8c15c95cbe9c2dde9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0306820bb9f4fb1dcda11c092a1535fd
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_a9e402efe5d574c1b690913b93b444de(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], -50, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_825d0f924de087f5ebab4ca9bcec636f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9e402efe5d574c1b690913b93b444de
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a558e0793c31ff91eacbba387bdfe25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95109bd6b94b41007c0e1f63000f70f3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_572c650ff72ba7e0d118751410bf2ca9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8fd2faeade559d77a96aa489df0cf0c
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_c42ea924c62569f1674b9193c5a808bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15006aad1c5611252c9215622d98664a
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_f633efdab7ab28ab6c0765b90d0bc2de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b77676a4bf5b87dd75109a82f0f76c1
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8c8afa339da46c3c4e981fe750f61605(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a4f96fdfa86d19cdd1c558cd6713c0e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_f14f66574127e13cb86738f88e6475a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9ba643916772e8424bc886e7c12e0c1
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_1a8204a39f0d0f1149f9bfcde90cd701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f684bc3beb79b493a511c02af321449e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_19ebdd27d8f55d774602a71fb3a20612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76ace68dab376620e9fb04790a7cf46b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a558e0793c31ff91eacbba387bdfe25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95109bd6b94b41007c0e1f63000f70f3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_a9a43d6440fc55d67f3bc7f55f532834(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 2, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_62a54fbf6fd74869903652360ba88133(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9a43d6440fc55d67f3bc7f55f532834
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_707994c62d8e5f9bde1795db6fd6b563(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 0.75, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_82d76f734ff31652c91f561c317a2d62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_707994c62d8e5f9bde1795db6fd6b563
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_3b10c31786c87ceac187ee5a4090a040(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ed82428d8739636d019f63588d48544
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_1a8204a39f0d0f1149f9bfcde90cd701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f684bc3beb79b493a511c02af321449e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_6a8b89d6ba67997a4f45599dfb6487fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91744f754c892adcead2511f1f945179
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_4e8e2ca8d37136a913360d7aa114cdfd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ecabc1f9dced2ff2145690045cbf6c8f
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_ab6e3f824e07eaa14216b70889b9b4c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59d428e94533601bc16671b552584b86
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_d7d1341dd97f4439fd14ad93bd403c8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_776dea60bffc5234342c42cfbf556a65
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a591d97808b801aa8b83999c48d59b7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b9bd69b9f40c9f943187eb6d703f689
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_df16027f4ff5afa84be01b9b472adf08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c682d387b92351f95aecae1088d034a2
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_9e26144c82841a0b7024534df87d9470(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e1765331d2cd655bd1b3b9efcad13b3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8bb29e2528dbddef14fa8be190b7ec6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8aab409109d28df903667e0731215e8b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_2056885415e89869e065d7bb3098b641(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 68, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a10581bcbad6bb5b3a89024a2c81d42b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2056885415e89869e065d7bb3098b641
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2309a2f8454e3f25ed99cebb50f7b2d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_364e0a5000b30cbe7447f3a7b48d65aa
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_97f2866b0bbf6cbe1cbc4af91a75bf83(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([4624, 1], 8, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3fae6d8720c28ea46cbd4ea7d25e72df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97f2866b0bbf6cbe1cbc4af91a75bf83
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_8c80bd4d82479b4f49ec736b72103e55(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 34, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4967264ffbe28d1b854e1ccde3a34d43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c80bd4d82479b4f49ec736b72103e55
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_553d5f29dc5959ffeb18466b22d6ff2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e5be3a4d141b05ca8e3c269ba9e2133
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_cfbeabfa37fa9d8d77c375181c14c3f4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1156, 1], 16, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e5cdc9d48842e4f5819cbd9a0347bfc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cfbeabfa37fa9d8d77c375181c14c3f4
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_0cd4a7990c6f749a0799cfb9d2659cb6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 17, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_70332d34581b47972281d7c0ab618c6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cd4a7990c6f749a0799cfb9d2659cb6
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_68a825a78a12c2b31f7340d146873488(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4143be70b6a83709b075d6203f2a5577
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_86c7eb29219d6b8f304e24d0f44b07ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([289, 1], 32, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_57fb885a1c131ef2dfa0875588b4b0f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86c7eb29219d6b8f304e24d0f44b07ef
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_1a8204a39f0d0f1149f9bfcde90cd701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f684bc3beb79b493a511c02af321449e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_0168b1e8a6ae8a04d43625ae2f3d2d52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7986a3d6d389630a91a1cc7dbb92b680
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_19ebdd27d8f55d774602a71fb3a20612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76ace68dab376620e9fb04790a7cf46b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_aeb6f161a8745db9fc1370a92ba6abb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5325d0b911c185f407d23d635ff07a4c
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_427ddceb6f78e3ee9dd6e4ca8cbe147b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db7cd1711a66290c002619165e161e7c
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_62a54fbf6fd74869903652360ba88133(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9a43d6440fc55d67f3bc7f55f532834
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_82d76f734ff31652c91f561c317a2d62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_707994c62d8e5f9bde1795db6fd6b563
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_3b10c31786c87ceac187ee5a4090a040(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ed82428d8739636d019f63588d48544
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_e84fd08983ea61db5f6e9d26aecd984c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f30156723f1493e28f82f7bcb842a6f
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_dd50d2e470c0219f783c065fdbac8978(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b082ff195d3c6b78320cc9ad3e1a27b6
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_73a3053012d555ab7565968b01343b91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4498732842f2d48432721196a11bb7a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_00b5fe9b425704b6a267f9fa46c51e6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ae06a6a28edd1d4ef7daa2193955670
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2c2593aa06a2163b3136a90856e2581c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2206545d093007dffd5065cf3c441627
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_cfa6f473ec7312afab0b33bdb2f0e438(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a941d081b4fe94e3cbad79efd4f2f9a6
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_827d102786d013b40bdd26b7ef63298e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a5da53036182f346c07ea5fc3ed8632
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_4e8fa2e227d494a3f6a8522b5905f72f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7209f99ccd37b7e49637ebaa32a52f80
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2309a2f8454e3f25ed99cebb50f7b2d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_364e0a5000b30cbe7447f3a7b48d65aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_553d5f29dc5959ffeb18466b22d6ff2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e5be3a4d141b05ca8e3c269ba9e2133
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_68a825a78a12c2b31f7340d146873488(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4143be70b6a83709b075d6203f2a5577
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_20df9b38cfac761e8fe7ef6b4f2dc78b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d46b754e8d80415852fa55350761b1fe
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b59595607c11a00c1b97164dd2c3ac97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d40c3580e303579fd03ed34bd13e10ec
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_c2f82a8e03caeafd0f4aca225a92ef4f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 320, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b84d0c585ed020bb4669e38402a794ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2f82a8e03caeafd0f4aca225a92ef4f
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2c2593aa06a2163b3136a90856e2581c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2206545d093007dffd5065cf3c441627
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_485dde1c74b3455de459aa6fbb8feed3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9d0e23085c4d0c96e3e881d0902c120
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_f633efdab7ab28ab6c0765b90d0bc2de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b77676a4bf5b87dd75109a82f0f76c1
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8c8afa339da46c3c4e981fe750f61605(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a4f96fdfa86d19cdd1c558cd6713c0e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_f14f66574127e13cb86738f88e6475a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9ba643916772e8424bc886e7c12e0c1
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_1a8204a39f0d0f1149f9bfcde90cd701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f684bc3beb79b493a511c02af321449e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_19ebdd27d8f55d774602a71fb3a20612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76ace68dab376620e9fb04790a7cf46b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_96e40d5b754a92c01327af4b7ef9006e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c0aca59885d2ef4005bfa5c6591331f
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b77bc9b2933d557d58d5888c1957ea70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0386ad510cd9c3c2af491cc475af7f34
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_aeb6f161a8745db9fc1370a92ba6abb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5325d0b911c185f407d23d635ff07a4c
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_427ddceb6f78e3ee9dd6e4ca8cbe147b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db7cd1711a66290c002619165e161e7c
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2309a2f8454e3f25ed99cebb50f7b2d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_364e0a5000b30cbe7447f3a7b48d65aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a558e0793c31ff91eacbba387bdfe25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95109bd6b94b41007c0e1f63000f70f3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_1a8204a39f0d0f1149f9bfcde90cd701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f684bc3beb79b493a511c02af321449e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a558e0793c31ff91eacbba387bdfe25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95109bd6b94b41007c0e1f63000f70f3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_68da5a667b1ad8fb056d46f1531b02ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e00ec737e48b8f3fbed7aebce240972f
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_65c3f176920416ee5fc0f9ce712b62c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55ea6b7c692f5ba212f0dcddd11c352a
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a558e0793c31ff91eacbba387bdfe25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95109bd6b94b41007c0e1f63000f70f3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_295950e8cc213d2872aa532c19b84827(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cada470e4cbab7e1218beff7167196ad
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_007fa23c4a97504cf8c38d11a8143cfd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 15, paddle.int32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9134a4d3f949d68b282ffd877ed1e4c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_007fa23c4a97504cf8c38d11a8143cfd
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_6894fb76506753d7d98ffcaa9ee881ab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 15, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b126ae3e4b38bf7ed2df7bd99bfe2176(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6894fb76506753d7d98ffcaa9ee881ab
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_421df1aac096369d29b9eea5ed2348a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95cbd657ccc07eabfec381ae4c44f778
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_68da5a667b1ad8fb056d46f1531b02ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e00ec737e48b8f3fbed7aebce240972f
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_65c3f176920416ee5fc0f9ce712b62c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55ea6b7c692f5ba212f0dcddd11c352a
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_6a8b89d6ba67997a4f45599dfb6487fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91744f754c892adcead2511f1f945179
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_0168b1e8a6ae8a04d43625ae2f3d2d52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7986a3d6d389630a91a1cc7dbb92b680
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_f633efdab7ab28ab6c0765b90d0bc2de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b77676a4bf5b87dd75109a82f0f76c1
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_c7b9494b29d2992c39fdd2190f2bc4ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e006158c8e80926f5ba717c1a521302
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_9e26144c82841a0b7024534df87d9470(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e1765331d2cd655bd1b3b9efcad13b3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a87152b3cd089e715310903b248e3d0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e7f8c962c4c4f3f50acb5310cf1334e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2c2593aa06a2163b3136a90856e2581c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2206545d093007dffd5065cf3c441627
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8bb29e2528dbddef14fa8be190b7ec6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8aab409109d28df903667e0731215e8b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_691e735f0b5f309d6a88dd59b3ae1054(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f0dff07192990077d6fb94aa22aa2f9
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_509ed8321457ade6de3d772f1fe18502(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5cf049d907e5d6670f2a060e1085f8d1
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_52622c67cc4b869e3a000dffd14fa806(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05652cd19cc06edb45e94fe36db0eff5
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_1a8204a39f0d0f1149f9bfcde90cd701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f684bc3beb79b493a511c02af321449e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_19ebdd27d8f55d774602a71fb3a20612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76ace68dab376620e9fb04790a7cf46b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_983e4cb517db58e614f66b50f7015f46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2d96170fc2f7a10d3377262b23f8e10
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2a751c4dbcd2a3004a15a6b8850010e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ace018338dc37152bde11588ad34efcb
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a591d97808b801aa8b83999c48d59b7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b9bd69b9f40c9f943187eb6d703f689
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_ab6e3f824e07eaa14216b70889b9b4c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59d428e94533601bc16671b552584b86
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_9e26144c82841a0b7024534df87d9470(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e1765331d2cd655bd1b3b9efcad13b3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_73a3053012d555ab7565968b01343b91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4498732842f2d48432721196a11bb7a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_1a8204a39f0d0f1149f9bfcde90cd701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f684bc3beb79b493a511c02af321449e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_19ebdd27d8f55d774602a71fb3a20612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76ace68dab376620e9fb04790a7cf46b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_983e4cb517db58e614f66b50f7015f46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2d96170fc2f7a10d3377262b23f8e10
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_1a8204a39f0d0f1149f9bfcde90cd701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f684bc3beb79b493a511c02af321449e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_19ebdd27d8f55d774602a71fb3a20612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76ace68dab376620e9fb04790a7cf46b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_983e4cb517db58e614f66b50f7015f46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2d96170fc2f7a10d3377262b23f8e10
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_20df9b38cfac761e8fe7ef6b4f2dc78b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d46b754e8d80415852fa55350761b1fe
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_414cbe2c49ddbc62cc270018556d3439(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_900902bf4b6bf411697f3e4135095bdc
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_0168b1e8a6ae8a04d43625ae2f3d2d52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7986a3d6d389630a91a1cc7dbb92b680
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_509ed8321457ade6de3d772f1fe18502(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5cf049d907e5d6670f2a060e1085f8d1
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_b65b7af6df757b9d9c6dbaa4b4994903(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 1.08108, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5c005c7d62176650caa99c92b4c3caff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b65b7af6df757b9d9c6dbaa4b4994903
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b808c69275a521c71244434f4c934cc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc236b766a8ba248dbfd88878591bc92
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_df16027f4ff5afa84be01b9b472adf08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c682d387b92351f95aecae1088d034a2
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a558e0793c31ff91eacbba387bdfe25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95109bd6b94b41007c0e1f63000f70f3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_897010487cc007ef1e9b64a8493d0e12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2fd28fc70cde4a4f1950368004d1221b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_96e40d5b754a92c01327af4b7ef9006e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c0aca59885d2ef4005bfa5c6591331f
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b77bc9b2933d557d58d5888c1957ea70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0386ad510cd9c3c2af491cc475af7f34
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_4d7b7244bc884b86649fb51f2a9b86ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4936ab50077760dead200ad28e5aab0
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a87152b3cd089e715310903b248e3d0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e7f8c962c4c4f3f50acb5310cf1334e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_f633efdab7ab28ab6c0765b90d0bc2de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b77676a4bf5b87dd75109a82f0f76c1
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a591d97808b801aa8b83999c48d59b7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b9bd69b9f40c9f943187eb6d703f689
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2c2593aa06a2163b3136a90856e2581c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2206545d093007dffd5065cf3c441627
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b77bc9b2933d557d58d5888c1957ea70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0386ad510cd9c3c2af491cc475af7f34
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_f633efdab7ab28ab6c0765b90d0bc2de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b77676a4bf5b87dd75109a82f0f76c1
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8c8afa339da46c3c4e981fe750f61605(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a4f96fdfa86d19cdd1c558cd6713c0e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_f14f66574127e13cb86738f88e6475a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9ba643916772e8424bc886e7c12e0c1
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_1a8204a39f0d0f1149f9bfcde90cd701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f684bc3beb79b493a511c02af321449e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_19ebdd27d8f55d774602a71fb3a20612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76ace68dab376620e9fb04790a7cf46b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_4e5b0fe8e55b1ce0048ec7b498a4b763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cf677087dd01579c4b6a4890caad9a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_aabcd7dcd2477b681b01226445ecb077(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64194509319894b799f6df30002a5cb
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8bb29e2528dbddef14fa8be190b7ec6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8aab409109d28df903667e0731215e8b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_1a8204a39f0d0f1149f9bfcde90cd701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f684bc3beb79b493a511c02af321449e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_aabcd7dcd2477b681b01226445ecb077(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64194509319894b799f6df30002a5cb
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8bb29e2528dbddef14fa8be190b7ec6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8aab409109d28df903667e0731215e8b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a0ac03445350f26835d65b207edb70ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f44c6866020e1c0100d135816a48333c
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a558e0793c31ff91eacbba387bdfe25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95109bd6b94b41007c0e1f63000f70f3
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_e0f57a4da54205417dbdaf35fbe6bc07(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 7, paddle.int32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b03f5f232af372ec9b59ea7d01f00733(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0f57a4da54205417dbdaf35fbe6bc07
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_f633efdab7ab28ab6c0765b90d0bc2de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b77676a4bf5b87dd75109a82f0f76c1
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_aabcd7dcd2477b681b01226445ecb077(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e64194509319894b799f6df30002a5cb
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_ab335bb7ca75cde65b87c6f234a71c68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca4ecbfb76554c3b640e22788c87fdae
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_1a8204a39f0d0f1149f9bfcde90cd701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f684bc3beb79b493a511c02af321449e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_bdab2e36df1a403309cf8e75d40d4fff(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 152, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3c4b87489db3acdba48281f5325d9357(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bdab2e36df1a403309cf8e75d40d4fff
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2309a2f8454e3f25ed99cebb50f7b2d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_364e0a5000b30cbe7447f3a7b48d65aa
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_636854f2cd8f3574a4b5e7680a3abc70(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 100, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3e8a0b8cf50725d322cc1b92d30b98ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_636854f2cd8f3574a4b5e7680a3abc70
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_8fb0f3023bad0e48d54fe0b2744aa869(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([15200, 1], 8, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_afb56bec4235b739acac9aec311dd42b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8fb0f3023bad0e48d54fe0b2744aa869
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_b8f9f9dbf529269beef1c830f364e625(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 76, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_93aaa429209c51f686f562b233434f93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8f9f9dbf529269beef1c830f364e625
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_553d5f29dc5959ffeb18466b22d6ff2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e5be3a4d141b05ca8e3c269ba9e2133
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_d8ab6e7305698cbdc4af7e98fcb1f450(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 50, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e9e33d10d9450a551e77b298ba9393c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8ab6e7305698cbdc4af7e98fcb1f450
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_5fa1d652b4eebda4ff3a680e2a098b5d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([3800, 1], 16, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_08a30b22cc0f318c1dbc88d2df5dede6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fa1d652b4eebda4ff3a680e2a098b5d
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_bbb576baf463bda468520c18381d2ea7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 38, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f3f0fa79ef07b702379055fcbd8f4e6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbb576baf463bda468520c18381d2ea7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_68a825a78a12c2b31f7340d146873488(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4143be70b6a83709b075d6203f2a5577
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_47e96292396799caab18e9b0ac5d937b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 25, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5e32459af9cace78ff211143ac5cfcf5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47e96292396799caab18e9b0ac5d937b
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_68bf2db4dc14e895d0c6aeb481a8ce74(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([950, 1], 32, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_dbcdebb1e5ce30370aa4ff3584344566(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_68bf2db4dc14e895d0c6aeb481a8ce74
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_6ed7daaf6132d4532bda1401e8c6edde(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 19, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_92cdcad0811a6833e0e01361a8a40fa3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ed7daaf6132d4532bda1401e8c6edde
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_20df9b38cfac761e8fe7ef6b4f2dc78b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d46b754e8d80415852fa55350761b1fe
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_7e75b09a7e4ecf2232b4ee5343a31bf6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 13, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d28fcac3352c2764ffd29bc5b299a632(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e75b09a7e4ecf2232b4ee5343a31bf6
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_391ac30e4a12bec3b55b77aeb4d75da9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([247, 1], 64, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_11e20d9b940555140801f8f906d2007e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_391ac30e4a12bec3b55b77aeb4d75da9
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8982c65bf1c175616519ac61228e5cb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_556f1f58ed5d61f407f0e1e1541b6b4f
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b59595607c11a00c1b97164dd2c3ac97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d40c3580e303579fd03ed34bd13e10ec
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_f83aa019c93626c6a270acdfd1546709(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 7, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7e00818fe3335e3e09fee52380ac27c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f83aa019c93626c6a270acdfd1546709
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_8414c4aa42aff1e7323da92a736940ca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([70, 1], 128, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_af30b5c3478b1b5010495599db4831dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8414c4aa42aff1e7323da92a736940ca
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_aeb6f161a8745db9fc1370a92ba6abb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5325d0b911c185f407d23d635ff07a4c
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_427ddceb6f78e3ee9dd6e4ca8cbe147b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db7cd1711a66290c002619165e161e7c
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_aeb6f161a8745db9fc1370a92ba6abb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5325d0b911c185f407d23d635ff07a4c
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_427ddceb6f78e3ee9dd6e4ca8cbe147b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db7cd1711a66290c002619165e161e7c
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_4e8e2ca8d37136a913360d7aa114cdfd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ecabc1f9dced2ff2145690045cbf6c8f
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_ab6e3f824e07eaa14216b70889b9b4c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59d428e94533601bc16671b552584b86
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_d7d1341dd97f4439fd14ad93bd403c8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_776dea60bffc5234342c42cfbf556a65
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a591d97808b801aa8b83999c48d59b7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b9bd69b9f40c9f943187eb6d703f689
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_df16027f4ff5afa84be01b9b472adf08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c682d387b92351f95aecae1088d034a2
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_3b10c31786c87ceac187ee5a4090a040(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ed82428d8739636d019f63588d48544
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b84d0c585ed020bb4669e38402a794ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2f82a8e03caeafd0f4aca225a92ef4f
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2c2593aa06a2163b3136a90856e2581c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2206545d093007dffd5065cf3c441627
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_287dfb6cf3bf07919ba50ac5fd420eec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_584d2eb8563b949b4ede75832dba55df
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_c42ea924c62569f1674b9193c5a808bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15006aad1c5611252c9215622d98664a
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_782f57a473d0098da16fe3baed7be4c3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([247, 81], 0, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9b48829bfd7331be1d1fe27a1d744ac5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_782f57a473d0098da16fe3baed7be4c3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_3926a2dc52ea7242a289193e0a6264fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e0c50876d98164601ccb8a967b37066
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_495b6c56be1097e9bfb0dde7cea81c8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdc6d042a8fba2640bb65cc7017d7074
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a591d97808b801aa8b83999c48d59b7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b9bd69b9f40c9f943187eb6d703f689
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b578e6231c9ab10277dee9851a40ac93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d22c0f2354835635eaf0fa0fd6c227f
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_308150ae4577ed08ed610f4be89d45dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc7e398a39cf700afdb757f289c34009
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_65ef72be708bae8104babbffa128a769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54a95fdf6878864a27a5fdbcb8e7ef25
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_287dfb6cf3bf07919ba50ac5fd420eec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_584d2eb8563b949b4ede75832dba55df
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_827d102786d013b40bdd26b7ef63298e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a5da53036182f346c07ea5fc3ed8632
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_4e8fa2e227d494a3f6a8522b5905f72f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7209f99ccd37b7e49637ebaa32a52f80
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_68a825a78a12c2b31f7340d146873488(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4143be70b6a83709b075d6203f2a5577
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_553d5f29dc5959ffeb18466b22d6ff2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e5be3a4d141b05ca8e3c269ba9e2133
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2309a2f8454e3f25ed99cebb50f7b2d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_364e0a5000b30cbe7447f3a7b48d65aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_6db247c81155d36cf7ab09cc14f3cd8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d4fa0aca5c08978162ef63efa409901
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a558e0793c31ff91eacbba387bdfe25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95109bd6b94b41007c0e1f63000f70f3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_3926a2dc52ea7242a289193e0a6264fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e0c50876d98164601ccb8a967b37066
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7728bdc486306943077ea31df1cefcfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79cd1c1d54ebd9f9c9bdac524eaa7920
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_4d7b7244bc884b86649fb51f2a9b86ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4936ab50077760dead200ad28e5aab0
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a87152b3cd089e715310903b248e3d0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e7f8c962c4c4f3f50acb5310cf1334e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_f633efdab7ab28ab6c0765b90d0bc2de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b77676a4bf5b87dd75109a82f0f76c1
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a591d97808b801aa8b83999c48d59b7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b9bd69b9f40c9f943187eb6d703f689
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2c2593aa06a2163b3136a90856e2581c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2206545d093007dffd5065cf3c441627
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b77bc9b2933d557d58d5888c1957ea70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0386ad510cd9c3c2af491cc475af7f34
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_dd50d2e470c0219f783c065fdbac8978(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b082ff195d3c6b78320cc9ad3e1a27b6
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_73a3053012d555ab7565968b01343b91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4498732842f2d48432721196a11bb7a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_1a8204a39f0d0f1149f9bfcde90cd701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f684bc3beb79b493a511c02af321449e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_d72feb46fdb19c5e09b4611870356fd0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([950, 81], 0, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e255ef0a53eac28bf34c4a1eafa01940(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d72feb46fdb19c5e09b4611870356fd0
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_3926a2dc52ea7242a289193e0a6264fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e0c50876d98164601ccb8a967b37066
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_495b6c56be1097e9bfb0dde7cea81c8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdc6d042a8fba2640bb65cc7017d7074
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_f633efdab7ab28ab6c0765b90d0bc2de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b77676a4bf5b87dd75109a82f0f76c1
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_e84fd08983ea61db5f6e9d26aecd984c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f30156723f1493e28f82f7bcb842a6f
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a591d97808b801aa8b83999c48d59b7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b9bd69b9f40c9f943187eb6d703f689
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_1d7a84d680533a5c5628823661102067(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 116, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6521d21659dffcce74604d0d20e186a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d7a84d680533a5c5628823661102067
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_9606110d8f6450e7744f4dd2137eb824(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b38dbe8018bb1bb11bb624adf0ea1ea
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b808c69275a521c71244434f4c934cc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc236b766a8ba248dbfd88878591bc92
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_df16027f4ff5afa84be01b9b472adf08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c682d387b92351f95aecae1088d034a2
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_baa835e5a468aacb3c3bb9f42a13ebf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b876a8e507a388fbf905fb293b76c7df
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_1a8204a39f0d0f1149f9bfcde90cd701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f684bc3beb79b493a511c02af321449e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_1a8204a39f0d0f1149f9bfcde90cd701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f684bc3beb79b493a511c02af321449e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_19ebdd27d8f55d774602a71fb3a20612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76ace68dab376620e9fb04790a7cf46b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_983e4cb517db58e614f66b50f7015f46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2d96170fc2f7a10d3377262b23f8e10
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a87152b3cd089e715310903b248e3d0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e7f8c962c4c4f3f50acb5310cf1334e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_c42ea924c62569f1674b9193c5a808bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15006aad1c5611252c9215622d98664a
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a87152b3cd089e715310903b248e3d0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e7f8c962c4c4f3f50acb5310cf1334e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_c42ea924c62569f1674b9193c5a808bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15006aad1c5611252c9215622d98664a
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_aeb6f161a8745db9fc1370a92ba6abb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5325d0b911c185f407d23d635ff07a4c
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_427ddceb6f78e3ee9dd6e4ca8cbe147b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db7cd1711a66290c002619165e161e7c
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_0168b1e8a6ae8a04d43625ae2f3d2d52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7986a3d6d389630a91a1cc7dbb92b680
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b84d0c585ed020bb4669e38402a794ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2f82a8e03caeafd0f4aca225a92ef4f
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_b2928c240b21276faca3785a86e1517b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([70, 80], 0, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2114ed29d2d52dc22c701e5b56ba05e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2928c240b21276faca3785a86e1517b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a558e0793c31ff91eacbba387bdfe25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95109bd6b94b41007c0e1f63000f70f3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_3926a2dc52ea7242a289193e0a6264fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e0c50876d98164601ccb8a967b37066
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7728bdc486306943077ea31df1cefcfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79cd1c1d54ebd9f9c9bdac524eaa7920
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8517bafd6e021a5b61451455f6f231dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_68134d372e1c79bfd8c2095d4516b8f9
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_1a8204a39f0d0f1149f9bfcde90cd701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f684bc3beb79b493a511c02af321449e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_691e735f0b5f309d6a88dd59b3ae1054(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f0dff07192990077d6fb94aa22aa2f9
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2c2593aa06a2163b3136a90856e2581c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2206545d093007dffd5065cf3c441627
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b976a44c2afed26fd64342d44292d9ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f3bf925c4d37ec56d06d4cc46ff1577
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_c42ea924c62569f1674b9193c5a808bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15006aad1c5611252c9215622d98664a
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_96e40d5b754a92c01327af4b7ef9006e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c0aca59885d2ef4005bfa5c6591331f
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b77bc9b2933d557d58d5888c1957ea70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0386ad510cd9c3c2af491cc475af7f34
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b84d0c585ed020bb4669e38402a794ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2f82a8e03caeafd0f4aca225a92ef4f
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_68a825a78a12c2b31f7340d146873488(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4143be70b6a83709b075d6203f2a5577
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_68da5a667b1ad8fb056d46f1531b02ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e00ec737e48b8f3fbed7aebce240972f
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_65c3f176920416ee5fc0f9ce712b62c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55ea6b7c692f5ba212f0dcddd11c352a
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_f633efdab7ab28ab6c0765b90d0bc2de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b77676a4bf5b87dd75109a82f0f76c1
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8c8afa339da46c3c4e981fe750f61605(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a4f96fdfa86d19cdd1c558cd6713c0e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_f14f66574127e13cb86738f88e6475a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9ba643916772e8424bc886e7c12e0c1
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_1a8204a39f0d0f1149f9bfcde90cd701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f684bc3beb79b493a511c02af321449e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_19ebdd27d8f55d774602a71fb3a20612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76ace68dab376620e9fb04790a7cf46b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_d30c38ce888186114da428dbca66a2be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6883b47cae0f46a2172898580cbe248d
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_573ae6762f0156635c0d579317fbfa22(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 72, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_02af5c36b47a1f583fc2d683d845d79f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_573ae6762f0156635c0d579317fbfa22
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2309a2f8454e3f25ed99cebb50f7b2d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_364e0a5000b30cbe7447f3a7b48d65aa
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_9c213750500afdf2f348baac6f3f5c2a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([5184, 1], 8, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_99abb7a54ef17edf56230b577aaaad3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c213750500afdf2f348baac6f3f5c2a
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_3e5ef945fb71b7c6c4c78d678f994350(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 36, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_39b588f885aa4fdbdf66bc0a905ee246(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e5ef945fb71b7c6c4c78d678f994350
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_553d5f29dc5959ffeb18466b22d6ff2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e5be3a4d141b05ca8e3c269ba9e2133
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_a9fc384d0d81f1fb84af33bfc6d91a20(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1296, 1], 16, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_24f20013d46a31b3aece95e8d6e1e8a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9fc384d0d81f1fb84af33bfc6d91a20
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_7b1b1682d5d3a34805ed39c156875834(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([1], 18, paddle.float32, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2e15b0ef2bfa48aa84a83310424fbfd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b1b1682d5d3a34805ed39c156875834
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_68a825a78a12c2b31f7340d146873488(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4143be70b6a83709b075d6203f2a5577
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_375a1bb40935eb4c2f0b999ba984fa17(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([324, 1], 32, paddle.float32, paddle.framework._current_expected_place())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_197b6450ae33e8fbe22226884cce2b94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_375a1bb40935eb4c2f0b999ba984fa17
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_1a8204a39f0d0f1149f9bfcde90cd701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f684bc3beb79b493a511c02af321449e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_0168b1e8a6ae8a04d43625ae2f3d2d52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7986a3d6d389630a91a1cc7dbb92b680
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_19ebdd27d8f55d774602a71fb3a20612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76ace68dab376620e9fb04790a7cf46b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_59884f606a020622bdd05fec914981cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8aab910f89c1b1f773daf7ae13e0507e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_691e735f0b5f309d6a88dd59b3ae1054(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f0dff07192990077d6fb94aa22aa2f9
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a558e0793c31ff91eacbba387bdfe25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95109bd6b94b41007c0e1f63000f70f3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_572c650ff72ba7e0d118751410bf2ca9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8fd2faeade559d77a96aa489df0cf0c
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_6a8b89d6ba67997a4f45599dfb6487fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91744f754c892adcead2511f1f945179
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_c42ea924c62569f1674b9193c5a808bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15006aad1c5611252c9215622d98664a
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_d30c38ce888186114da428dbca66a2be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6883b47cae0f46a2172898580cbe248d
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_f633efdab7ab28ab6c0765b90d0bc2de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b77676a4bf5b87dd75109a82f0f76c1
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_c7b9494b29d2992c39fdd2190f2bc4ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e006158c8e80926f5ba717c1a521302
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_1566d0da727dc0b417f0a6449bad4860(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_179832a90b4a3da398a66868e37ff706
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a87152b3cd089e715310903b248e3d0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e7f8c962c4c4f3f50acb5310cf1334e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2c2593aa06a2163b3136a90856e2581c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2206545d093007dffd5065cf3c441627
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_65c3f176920416ee5fc0f9ce712b62c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55ea6b7c692f5ba212f0dcddd11c352a
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_e13f875b3cb7da10ff338fc779ce8158(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8b26ad00a514dfeff3b7c530feb5e6e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_3f17ebbc2c2865a8c15c95cbe9c2dde9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0306820bb9f4fb1dcda11c092a1535fd
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_aeb6f161a8745db9fc1370a92ba6abb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5325d0b911c185f407d23d635ff07a4c
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_427ddceb6f78e3ee9dd6e4ca8cbe147b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db7cd1711a66290c002619165e161e7c
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_ab335bb7ca75cde65b87c6f234a71c68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca4ecbfb76554c3b640e22788c87fdae
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_1a8204a39f0d0f1149f9bfcde90cd701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f684bc3beb79b493a511c02af321449e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_287dfb6cf3bf07919ba50ac5fd420eec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_584d2eb8563b949b4ede75832dba55df
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_aeb6f161a8745db9fc1370a92ba6abb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5325d0b911c185f407d23d635ff07a4c
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_427ddceb6f78e3ee9dd6e4ca8cbe147b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db7cd1711a66290c002619165e161e7c
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_ab335bb7ca75cde65b87c6f234a71c68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca4ecbfb76554c3b640e22788c87fdae
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_1a8204a39f0d0f1149f9bfcde90cd701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f684bc3beb79b493a511c02af321449e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_6a8b89d6ba67997a4f45599dfb6487fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91744f754c892adcead2511f1f945179
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_485dde1c74b3455de459aa6fbb8feed3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9d0e23085c4d0c96e3e881d0902c120
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a591d97808b801aa8b83999c48d59b7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b9bd69b9f40c9f943187eb6d703f689
    def get_inputs(self):
        return [
        ]


class PrimitiveOp_d9ba45816fb19ae5a27fab0a47edff24(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, ):
        return paddle._C_ops.full([], 58, paddle.int64, paddle.core.CPUPlace())

    def get_input_spec(self):
        return [
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d603d26975aba148b3ac6e9b28d65bc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9ba45816fb19ae5a27fab0a47edff24
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_6521d21659dffcce74604d0d20e186a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d7a84d680533a5c5628823661102067
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_cfa6f473ec7312afab0b33bdb2f0e438(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a941d081b4fe94e3cbad79efd4f2f9a6
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a558e0793c31ff91eacbba387bdfe25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95109bd6b94b41007c0e1f63000f70f3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_1a8204a39f0d0f1149f9bfcde90cd701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f684bc3beb79b493a511c02af321449e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_19ebdd27d8f55d774602a71fb3a20612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76ace68dab376620e9fb04790a7cf46b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_983e4cb517db58e614f66b50f7015f46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2d96170fc2f7a10d3377262b23f8e10
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_f633efdab7ab28ab6c0765b90d0bc2de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b77676a4bf5b87dd75109a82f0f76c1
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_c7b9494b29d2992c39fdd2190f2bc4ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e006158c8e80926f5ba717c1a521302
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_9e26144c82841a0b7024534df87d9470(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e1765331d2cd655bd1b3b9efcad13b3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a87152b3cd089e715310903b248e3d0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e7f8c962c4c4f3f50acb5310cf1334e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2c2593aa06a2163b3136a90856e2581c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2206545d093007dffd5065cf3c441627
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8bb29e2528dbddef14fa8be190b7ec6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8aab409109d28df903667e0731215e8b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a87152b3cd089e715310903b248e3d0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e7f8c962c4c4f3f50acb5310cf1334e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2c2593aa06a2163b3136a90856e2581c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2206545d093007dffd5065cf3c441627
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_a558e0793c31ff91eacbba387bdfe25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95109bd6b94b41007c0e1f63000f70f3
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_572c650ff72ba7e0d118751410bf2ca9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8fd2faeade559d77a96aa489df0cf0c
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_827d102786d013b40bdd26b7ef63298e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a5da53036182f346c07ea5fc3ed8632
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_4e8fa2e227d494a3f6a8522b5905f72f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7209f99ccd37b7e49637ebaa32a52f80
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_68a825a78a12c2b31f7340d146873488(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4143be70b6a83709b075d6203f2a5577
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2bd83f76d2c1b3bcd02cea5d8207a3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e2c516fe72354f0c1b3cf5f32acc0aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_553d5f29dc5959ffeb18466b22d6ff2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e5be3a4d141b05ca8e3c269ba9e2133
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_2309a2f8454e3f25ed99cebb50f7b2d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_364e0a5000b30cbe7447f3a7b48d65aa
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_ab335bb7ca75cde65b87c6f234a71c68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca4ecbfb76554c3b640e22788c87fdae
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_1a8204a39f0d0f1149f9bfcde90cd701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f684bc3beb79b493a511c02af321449e
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b808c69275a521c71244434f4c934cc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc236b766a8ba248dbfd88878591bc92
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_df16027f4ff5afa84be01b9b472adf08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c682d387b92351f95aecae1088d034a2
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_c42ea924c62569f1674b9193c5a808bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15006aad1c5611252c9215622d98664a
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_91010731da2403760e7ec22ecdd37767(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_869f62127d7a1b1c07ad3726a993b605
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_5aa60a6734876948033ab5eaab1dfc5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ca0d01d21891e1a2a6d65d62e64d91b
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_8176f3e03223726594f65edf6f85e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b0fbd77620e6a5bf14a53a72c5f4a7
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_6a8b89d6ba67997a4f45599dfb6487fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91744f754c892adcead2511f1f945179
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_7261ca5970057baf7af3a4448bf6763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fa5af80083259b5bd8ea88a5b23f139
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_b105dcce48cff45200fa807bbd021b64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4155a712ef006bed9a4cd9a779378da4
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_64a9b9a1100a4fe72f969bd279a55f99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53574864b5f186c8c174c17bbd04dbbf
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_d30c38ce888186114da428dbca66a2be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6883b47cae0f46a2172898580cbe248d
    def get_inputs(self):
        return [
        ]


class TestPrimitiveOp_485dde1c74b3455de459aa6fbb8feed3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9d0e23085c4d0c96e3e881d0902c120
    def get_inputs(self):
        return [
        ]




if __name__ == '__main__':
    unittest.main()