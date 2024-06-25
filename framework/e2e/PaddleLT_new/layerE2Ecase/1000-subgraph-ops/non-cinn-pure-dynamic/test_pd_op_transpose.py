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



class PrimitiveOp_ec48a690c1f8119f96b9b20832556439(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_12cb9eeeaba75c523c07c3f282bee05b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f77bcdc90d2aabbf93e8b893de2b814e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99e35bc82e24367d51e280f0853a2834(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 1024], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5c0231f5135d9d82553cda0bc0e5534a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 6, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_938450b0d47ac7a4342ef4aeff19345a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c565d4dd07791abcaab8b0fc2655359(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 49], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 3, 1, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e61e00430f37ed8a99145862bf94e638(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 2, 6, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ac61136039a5b7e5e55e87930b435ce8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_fe82cab8d58e76c293086a19456b745b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 3, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8c30deed42a501e7978bf867c7ef1d8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 168, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e48899fbad2d07a1a947a939108d15ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5027d073840d8f7a40c21467294ffbcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2997981ab8eb4010a497bf658dbd5ca3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99a857f824c9b79ad5ebdf5f4e49dfb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_58633005c3de23934be5eb27a376c920(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 168, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c15bead3b0983bc2188c6c96ede9e341(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 84, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_018ab25c6d2b30777f050caf9170858b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 42, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_93370e0ea909d0e8f51b29d639c9f921(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 21, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3f2938cb2a80526df3807bc57ed5b63a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 11, 16], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ff3b4a1a7327cf7244031390082a7911(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[2, 0, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c1faf881e36c9693a1275cbc7c1a8f02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff3b4a1a7327cf7244031390082a7911
    def get_inputs(self):
        return [
            paddle.uniform([300, 256, 49], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 3, 2, 4, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9c93838738d534d816cae3c7d09336e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([11, 8, 7, 8, 7, 96], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_9d86c553350cb977b271d6b77a7c3701(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[3, 0, 1, 4, 2, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_38e78d3f8ab86c8931b19c3233330808(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d86c553350cb977b271d6b77a7c3701
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 49, 3, 3, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c81b4cc87754d47601950d8e7ad403bc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 1, 2, 4, 3])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_281ce7c83768d01e0f68771c7469a448(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c81b4cc87754d47601950d8e7ad403bc
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_109accd2f7e47a053daf8450bc106061(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 3, 4, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fb2308262882fa1cacbc0966f6cd4d8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_df853f2a9b6fd4cb181955826b127189(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd1607756441eaaae2af8d8953a6498e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([54, 198, 3, 3, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cd1d1c758f26e73d61561d5e662e62bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 198, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1ef2739079786eaa92a481d61c0c8bb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 198, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_72d347787330027aabe51b9d6385acb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([1960, 16, 2, 4, 6], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ec7d16e7d0122ea868f80ca32670e5d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1960, 16, 4, 6], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5e9299db047f2626eb2bf6e1b678a5a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([1960, 4, 16, 6], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_727578ac1fc6a4a165d47da6fc216506(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 6, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5bdd3d31a98dd758df7452f070f5ee7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_641c61f628794dd566e19148b0f1de01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 49], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_edba97be2f842d59217e50ded2eb7b15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 2, 6, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_44d161b0edfd9361767b64bca461c4d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_c93a70902343a6481859d47a82215837(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e1ecc5bb2385d9525968235095a0dda0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c93a70902343a6481859d47a82215837
    def get_inputs(self):
        return [
            paddle.uniform([16, 32, 128, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bbde156eb6d2afd9aca398c6f4bcc29e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([16, 128, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_89f5b09f5382f5c70693d7e08178af54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 7056], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_53da85cf50f209df316b1a9dff7c7599(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 7056], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a1962218552c892284351deae0d9d9fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([43, 8, 7, 8, 7, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_be17daa055ad210c54b7ef1edbbd7b13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d86c553350cb977b271d6b77a7c3701
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 49, 3, 3, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_acae64e08512b6fff834c54d8521e90d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c81b4cc87754d47601950d8e7ad403bc
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_78730d51b5fd6e88ee808d5adf618c0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c93a70902343a6481859d47a82215837
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7674c1698f56d33ae05d6037e2cfe940(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 160, 240], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3a502bb7f358f4d017647c6e4024221e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 80, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a03925080efd1c07e2e9c335f4e07a1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 40, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6a9f60b1f0f536fb185e8f87d05c8fb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 20, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b1566f2db52638a24187c21bd1d8b4e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_08fbad537f2834885885545ff7edda15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 160, 240], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9880f6840b8d164b101a360ee4e88099(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 80, 120], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e744056a668a9b60215d0f50d2ea1bfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 40, 60], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_17d579c3b527b290077a83b0234c23f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 20, 30], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bc61474b33085507d82e71211c1e639(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 10, 15], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d3780f7e22c705ce471677309bc835eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_620b2a04405c5ca11382fc0cf1a872c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fd28c5505515037acb53cf00a0284056(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7911c70783b0f03d8796e4b8ccbdfaab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 576], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7b4163d97d42313db37ba31f3a97d02f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 576], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1600de5680154a729aee017fea8878c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 2304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e7700bcc7c82e7517f78d26dcbf880f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 2304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c9f3b976e0c1f0c6c4f6cba30ebe3f51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 225], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_205e707c82eb546efe2dea0889db76f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 225], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6100e81808992d50e78e9c7ecbba9aef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff3b4a1a7327cf7244031390082a7911
    def get_inputs(self):
        return [
            paddle.uniform([100, 256, 49], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e6923c7c04758b9b93d470c599b9855c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 2, 1, 3, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e330544749c9c3dbd33751b208084bd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6923c7c04758b9b93d470c599b9855c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 20, 128, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4219fd5257ff3e96ef569965ffa20cf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6923c7c04758b9b93d470c599b9855c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 40, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_528e5479fe88a2593837b38c5a0b3e15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 152, 272], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3b6d318263f903b35f6b83ea93145aca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a86201b8325869840fca2f0b10cb1137(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d60fe5f3f059e1087b7d289e69cbaa05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c36f4f856098a45a74670983eb0b5be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 12, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_67c83f0b2e82337e5045c38f0b02e762(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_89e3e90d5a8be2d10c5ec0044db458a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 49], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d014172ebecc5e740716c6d0358be713(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 2, 12, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_58bfd1740e7798cca15c652ac2d1e5ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([43, 12, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_74eebdf8314b6deea075dbfaa877c8e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c93a70902343a6481859d47a82215837
    def get_inputs(self):
        return [
            paddle.uniform([128, 16, 8, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7b24e38d1da22cdca83e0fca2b6fd387(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([128, 320, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_803b89292a95d11cc342725eb722640e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a86201b8325869840fca2f0b10cb1137(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_efbcd7d82476d7bd965eaa2c1b1f9977(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bd4d23f636ae6a537492c2bd14ea8d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 676], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7c82b9d03745c0727e9639b6ef685b39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 76, 676], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_080aad177e596ec10ab8aa5fc00f462d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 21, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a12f8626afa741b5bdee8fdf1f61341(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 120, 216], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_ea3c04f43d1994a7ab54cbf907ac684e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 4, 5, 3, 1, 2])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bb7280a086da7a7372e251c1cb25f820(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea3c04f43d1994a7ab54cbf907ac684e
    def get_inputs(self):
        return [
            paddle.uniform([4, 8, 8, 128, 13, 13], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1a530fb03eb1f2cff304154104f8acee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 900], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_166bacb3abd7d7343b21a2f4b76a98e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 900], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_08b1ebc52bdff1c3ef0b3564c7912882(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 2704], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1ab916d93574d6adc9d168a18fe371e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 2704], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_59afb321b84943fc61f938be414876ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 3, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0f8fe4bba876031b5f6a56970f64115(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e6ccf5afd22cc3fa1c758d5e91f0861(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 49], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8448577a18bb927180ef0298c5551140(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 2, 3, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4c4b1e931543a5408295c7d89d2d4908(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([43, 3, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f8ab3ad124a49d2636cc5c9e3d2516b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 3, 4, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a3e3e3ba86e2730e2cbc13f19e056b6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a8591be6adcedce1b6b1c841be77eb7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6adbd897d1b297657594f8334b351653(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_20ab57c76260eae484b7121ef1d55fd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 169], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5548ada659cecfe3684e22e66aa8aba8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 76, 169], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0ec037eca46d14957b72f4c9aadf7dad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 32768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4efab8424e103932864ff59eefb84d07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_835501c42ba1b3258f435264e1ab639b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 16384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc3be67777734e4973c0f37908aa6765(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 16384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_79a5522489b82171d22f262f5cd4e078(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 16384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8157316e707a575559dacc4864144a01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1156], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_337364fc71a562476a4c82b7da31038f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 1156], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f784291996b73f18896bedeb13c6368e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c93a70902343a6481859d47a82215837
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e18d419020e51a652d0fb0c260b429b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([528, 4, 96, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6260ee352d36ff16bbbf4a74e93341ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 24, 48, 2, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f8465589d9eac338cb0e63bc0dcf3b49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 5776], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eb294e9098759753dc73f6b52baca70b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 5776], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5fafb6f77a6ade405b2096da7284c66e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 7, 4, 7, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c74a72ca485e66200fd8b5feac3246d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d86c553350cb977b271d6b77a7c3701
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 49, 3, 6, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3fc662472882b37a976cf357cd9c4015(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c81b4cc87754d47601950d8e7ad403bc
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_537a37dea364391b58f92915365c5c7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 16384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d75c7c1eb40ce103e2be394f6268e358(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 21, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7911c70783b0f03d8796e4b8ccbdfaab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 576], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ef573b677076f5c4cdc3760c2373e5d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 576], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_13a5b5cf9b9cdf7cb1ec49de88513aec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 576], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_add83e19128816ed2122722bdbe497b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([6, 2, 1, 12, 24, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bbb4139650d1feed13b5c8f3c06b9f4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c93a70902343a6481859d47a82215837
    def get_inputs(self):
        return [
            paddle.uniform([8, 16, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7d309a44bd1b4d0e2882b98052fc14a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([8, 320, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2c4ea860586d7ca47952e090f19431cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c93a70902343a6481859d47a82215837
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8f4c01072783f0190e57d4f46014b751(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c93a70902343a6481859d47a82215837
    def get_inputs(self):
        return [
            paddle.uniform([8, 16, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_11a236ed98401265ca3d59c77af4dd93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([8, 160, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7d6d5160ec6117b57fdb37f9234a60f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8384477e5201f0745674414370cb078d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 3, 12, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06d3e7065af020160c170478d625a33a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 577, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e9862cb3cfb220169c308c48e33ce5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 577, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_85558f148ef6c66ccb44f66b639e4ca2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1296], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_591a5a82a27bcb9fe5eea3c0722d0da4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1296], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a2a61ac545de702bc05b0008d800cc8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 1296], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8c1869de921f4b585926fc2a225d12b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c93a70902343a6481859d47a82215837
    def get_inputs(self):
        return [
            paddle.uniform([64, 64, 16, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1a6da94a3c617fd696ac44abb78a44a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([64, 64, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e624f5ebfe0c9e92af846225be702039(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([10, 197, 2, 6, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ef3436c1132d859b1cb1393774117039(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([10, 197, 6, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b97da2a58557a9762a1bb0a865a125d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([10, 6, 197, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_726dce5ecbb4a72c47293441f2b93576(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aaba694b76b16c9763c0a4b3e5284f59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([384, 2, 96, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b355dd56782867b23f846deb057581bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([4, 1, 96, 96, 1, 48], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8b57d4bff7645e2abf996d5cdd87242c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 7, 4, 7, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_35c8892a170503a7ba146959f7e2f73f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d86c553350cb977b271d6b77a7c3701
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 49, 3, 6, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36b5543b66e84ada2658ce63b78c793a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c81b4cc87754d47601950d8e7ad403bc
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f8339bbe8958582d138a087683e59b77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 16384, 2, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_80faa545bd62f5fc43125d1c8ca53ee0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 16384, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8e1cfd6b90e78207bafd4a88737edf8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5fb77fe1d9058816c0b59df36945df51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 2, 2, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b07be06b59b3a12225b09a8d7f4f799c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 1024, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_747f906bd642153727f05328a72afac4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16384, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_803b89292a95d11cc342725eb722640e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_59afb321b84943fc61f938be414876ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 3, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0f8fe4bba876031b5f6a56970f64115(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e6ccf5afd22cc3fa1c758d5e91f0861(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 49], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8448577a18bb927180ef0298c5551140(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 2, 3, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4c4b1e931543a5408295c7d89d2d4908(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([43, 3, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c7ea731fb477c8679acd73f5ea90eec5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c93a70902343a6481859d47a82215837
    def get_inputs(self):
        return [
            paddle.uniform([16, 32, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8594cb92d2130d9c82d4619589e58c8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([16, 128, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_20ab57c76260eae484b7121ef1d55fd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 169], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_246557cd47082a376e0a70a5dffff7fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 169], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3b4dfb9cb234b7e4821277621dfe65d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 24, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_85c4daa45134660309045eb5b2a5cd63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 2, 24, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5a734ce90e791704b1d6b58824f00f0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([43, 24, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e851e8053f2f6a939dec84d0f4856fed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_57a5518a0b03921036060e7cb76bc12d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c93a70902343a6481859d47a82215837
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2febe4f018691c0830927e5ab69927e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 400], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1a74bd79542132c504ace3126bed75d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 400], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_34c83936933271e934f9a6a56565a9f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c93a70902343a6481859d47a82215837
    def get_inputs(self):
        return [
            paddle.uniform([8, 32, 64, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3df64db947bb1aed7d8caf11c30d08d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([8, 160, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cb2f6487f7b90225bad60cb36ecc97a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bb518271503f703305b2aa3a12c47a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c93a70902343a6481859d47a82215837
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e93bd446964b429f2f678ea1147319b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b95d366e31617858fcf4a555b47c08ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 60800], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8fd03f611e23dd736bb4b4fe93437535(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 60800, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b95d366e31617858fcf4a555b47c08ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 60800], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f5fbf9437c9bfb3478de0cf6640b9b2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([10, 640, 3, 2, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0e23ea349378d43676955b805a88b7eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 640, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_65acf5301cb612bcdf0ff3fcd2bc588a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 640, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_15112fb591040991bd8130e1ae79617a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fcce158249452e0d90ef7cc75cc3a393(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([86, 198, 3, 3, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f057543f504ed5e456eb4d0c4a1ff827(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 198, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_782e74e4eeaf30ad34ad072d499f49e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 198, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e57e7e8d501ca280d45030b96f4b9464(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1600], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_95a0c199180bad02ff5a2efc8c924831(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 1600], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f2d1f304f0d3b3da5491482fb2b14c96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 3, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e8ea21ea31c94482eee0b12342c555e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b2e2b169b94aeaca64d03fc79fb84663(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 49], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6f611c801d69db458480d96e35ef5f3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 2, 3, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c508cf6a806c60b3f7ab0aaa0de3d568(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([11, 3, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1241e26afbb7971bb150f16c50d05b25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d0ac3a7653fde012a080a9e79444a826(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([20, 8, 288, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_426a32b633f64c18bc73fa565867f191(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 2, 24, 12, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5a15e3420534fe0ef0e26a912a292d0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 196], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_59de44dd933a229002df58eb43304d24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 196], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e29e9d891003950644688c23ffe47698(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 7, 1, 7, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_522a491dbe8b7a562a6bc0bd7ebaa1a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d86c553350cb977b271d6b77a7c3701
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 49, 3, 24, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_419bfe7d76953289487325884f1a7bae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c81b4cc87754d47601950d8e7ad403bc
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_be0187dbff0ba6c3c87f9c1c692d61f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([4312, 16, 2, 4, 6], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_399627e22aa4e61285fe8e46d9b645c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([4312, 16, 4, 6], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9ae96cc7145f4328dc56a0768eb2d272(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([4312, 4, 16, 6], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1600de5680154a729aee017fea8878c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 2304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e7700bcc7c82e7517f78d26dcbf880f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 2304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_222311105b08ba858064e21514e24bd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 441], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_27918c59e56de2cb50972e9c6d95d61e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 441], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a1962218552c892284351deae0d9d9fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([43, 8, 7, 8, 7, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_be17daa055ad210c54b7ef1edbbd7b13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d86c553350cb977b271d6b77a7c3701
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 49, 3, 3, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_acae64e08512b6fff834c54d8521e90d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c81b4cc87754d47601950d8e7ad403bc
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8157316e707a575559dacc4864144a01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1156], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b2e18a5d5955a90dbaeba78ca9ba4e4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1156], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ee8c3ee248b314e3ad946c8e7b36a07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 1156], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8bf59add2fbd4a8a04ccfa87aac6fda0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 1280], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8e16961373b1c103fddc18723f89ac29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1280, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f186441ad77b02b7cbe4b765910ff683(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 176, 264], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05492af57aa4e8833a5eb82a20bb3d75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 88, 132], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f7bd59253748cee288637ef694b0824b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 66], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f3ae66c2fdfc38a4b7692da4309f9fb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 33], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99a857f824c9b79ad5ebdf5f4e49dfb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fa9e492d5211eabeb361d9726a7355b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 176, 264], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_912d1a5171132cad24e4c0663de790fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 88, 132], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_589e4c466c6c66857a03e4d10855f8a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 44, 66], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cedef72ce0c30989c92a84c2678e8538(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 22, 33], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3f2938cb2a80526df3807bc57ed5b63a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 11, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_70c2f5641106bcb8050bd5566f61219d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 65536], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1a564f4c3488c8517220616c7d7d2291(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([576, 2, 96, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_94df1faeaa1763bb190aaf375cd6f573(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([6, 96, 1, 1, 96, 48], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_164d3ffccfdb05eef285ca900a3ac5ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 324], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bbcf70d30c7261997ce46f3c683b831f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 324], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_98148b46af67598e98e4a11b143dd522(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 324], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_047ff5e5df3628376b3a11de73e4851f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7e92d6637c75563f85110d3a916d1a0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 289], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_86b84c58e9069f3d7663f459ff479648(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 289], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_89b05c39bd8154d30dd7a9f1fdb95ac4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 289], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0933ac2b44d80f703444f0711a682460(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([6, 96, 9216], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5fe2ba54c73cd7db30cf880b4410230a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 144, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0389a28a83e1d9ef542eee57793133e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 1, 12, 12, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_820285a776d6fc9b42e52fc828fd04e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([96, 4, 96, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ecc4a642b0af5c4b29cd97b317716670(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([4, 1, 24, 48, 2, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_540920b298a3f9b5c356fcef0f397af2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([12, 8, 288, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_add83e19128816ed2122722bdbe497b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([6, 2, 1, 12, 24, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_da6a5698dd42b2ca46d8483f05deae25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 8, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a52174a0a0fe750cfb37ee3346c2ab17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 2, 8, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d154c48caf0021afcfaf7aebe62e8584(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_667f2f63b856eb21c07e7bd8cec6815e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_93f0b915a85c3bbeef6fd90449cbc934(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 3136], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_052ffacaa912c24b5ca17cd50a67e1a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 3136], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba22a887e0bc4cdbd4e64a8171caaa6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([6, 32, 144, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_79cfb6f33f40b1044a09f210aa1111a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([6, 1, 1, 12, 12, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_803b89292a95d11cc342725eb722640e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a86201b8325869840fca2f0b10cb1137(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_efbcd7d82476d7bd965eaa2c1b1f9977(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_af1a9fffc13248b2016a34ad8fcd6959(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 196], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_59de44dd933a229002df58eb43304d24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 196], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_97c20c1a19a62b9fbda514c2e694d53f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 12, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c6de3d33e41e1a63a721541715d6402a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c7db052e1af3bf3bc6e1f7dbb9cea76b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 49], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f021a6a86ea229f3545f42afdb009898(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 2, 12, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_63b7363c9e972009a1984169f699ef42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([11, 12, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_af41929ef26d5b473e5630a4a43ea098(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([960, 2, 96, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e71df91e0238ec3945cef8449255b202(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([10, 96, 1, 1, 96, 48], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_109accd2f7e47a053daf8450bc106061(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 3, 4, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fb2308262882fa1cacbc0966f6cd4d8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_df853f2a9b6fd4cb181955826b127189(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2789b37afc615743a862849338de96f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 38, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_51acfb43fb9a42114f575bedc0878973(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 38, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a3800658e58bc0fd9d1fba590f0df3c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 19, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2ac889a92747570501e81de7edaa58de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 126, 19, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0f81f41e0b6a87a8fcc054fb1bbe50ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 10, 10], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_095da5c37cc7dde198883c61f2784b0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 126, 10, 10], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0e6cf4e67d815db35de586d3b627937(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 5, 5], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7ac4877c0477658a07e4b44ea7d133d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 126, 5, 5], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_985b46a575d8ccf9373192dbd6d4b773(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f04cb33cf19cc175952223cf2245aa6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 3, 3], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f885d2111dbc4dcbfffc2cb1879e9537(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[16.395017623901367]], [[14.621191024780273]], [[14.839827537536621]], [[14.212995529174805]], [[15.577754974365234]], [[16.321489334106445]], [[15.02302074432373]], [[15.390174865722656]], [[14.966716766357422]], [[16.038286209106445]], [[14.679195404052734]], [[15.207182884216309]], [[15.759089469909668]], [[15.713208198547363]], [[15.664520263671875]], [[15.50053882598877]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


class TestPrimitiveOp_2f0d66aea7cbf999726b60825cda6be4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0286a62cb6d12805d3e45b90ff253e89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([2112, 2, 96, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9d94c4a879052de9f88885d21275037b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([22, 96, 1, 1, 96, 48], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_797e8cfcf8de6775bf6dfdcf02b21a27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6923c7c04758b9b93d470c599b9855c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 36, 28, 50], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_208865b162d87275ddc59e838f2865a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c93a70902343a6481859d47a82215837
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7e92d6637c75563f85110d3a916d1a0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 289], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9ce455145c35f621942f61253e5c9424(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 289], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4796c4cdb0637a4db7954df317c13d44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([22, 49, 8, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4796c4cdb0637a4db7954df317c13d44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([22, 49, 8, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d2f90c4cca67bd193154eda91b3bfcf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([22, 49, 8, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dacfe02a6048306fd859f868b97db428(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 49, 16], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_1006b823370d28fe0b4e283c48eeec6b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[1, 0])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9e34ce247f7d679cba953219e019d5d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1006b823370d28fe0b4e283c48eeec6b
    def get_inputs(self):
        return [
            paddle.uniform([8, 49], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5ecbef36cd67fc49e157f35ab0850d2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1006b823370d28fe0b4e283c48eeec6b
    def get_inputs(self):
        return [
            paddle.uniform([2401, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5bdd3d31a98dd758df7452f070f5ee7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_113e5838dffc656774b8842c7041ed81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 784], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_67c83f0b2e82337e5045c38f0b02e762(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_323bba8b922512a6cf48d51648fd28bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 196], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_295917a100b27b1c4633e5850dc907df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 784], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_01c6e4e61074e5cb5f635e71c56ae3cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c93a70902343a6481859d47a82215837
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_e256fa5e4115d3f738da9fa362d74ffb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 5, 1, 2, 4])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_01e1865282939e50e028397fd6b8b5af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e256fa5e4115d3f738da9fa362d74ffb
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 4, 8, 16, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8b57d4bff7645e2abf996d5cdd87242c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 7, 4, 7, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_35c8892a170503a7ba146959f7e2f73f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d86c553350cb977b271d6b77a7c3701
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 49, 3, 6, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_36b5543b66e84ada2658ce63b78c793a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c81b4cc87754d47601950d8e7ad403bc
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a058343fd8fe89c7402060a6190d9d5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e256fa5e4115d3f738da9fa362d74ffb
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 52, 8, 202, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc111045f27785b757a99fe2930f4039(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 200, 304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_70e552f61836ecaec9f1b2faa948c29a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 100, 152], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b33792ef44d17d2cbcefa64e12cc2d4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 50, 76], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_60555da1d04c5201b1d2d254c7515226(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 25, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aa43fd72a5ff301f42d7a9a05d306d32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 13, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a2170ab527fda500d5274e17cf755071(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 200, 304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_12b40fab346607578c9954341a5ec16b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 100, 152], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b5bf993019f8ca6f128a62ea2239df33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 50, 76], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_157141dfb0aac26bdae6d998d2c9a258(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 25, 38], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bee2d9897f9159d03aab9785201ff4c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 13, 19], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_27629cc3e07bbb150cc5fcb93cd33d56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2116], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_70feffe64d121c205603f48b605d176f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 2116], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_12cb9eeeaba75c523c07c3f282bee05b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f77bcdc90d2aabbf93e8b893de2b814e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99e35bc82e24367d51e280f0853a2834(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_45434c62dc8fbd1d022c86912dc1eee9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 3, 6, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d3424969311417cc3ee7dba7e60ac4fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1025, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_38d715a1965dd29c8752a14faaa4cd2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1025, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e591176aadbaa6a7ee6e0915d2b7f553(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d46851e251e613a477646f450b416a59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2ca6a65cffb9e7ef00a8fd510d207320(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([22, 196, 8, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a2e6ff60b8ef4ba9515bba47943ed7eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([22, 196, 8, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_491226e6b86b40afae5f2b41a9d045b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 24, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_82889a7f391abf28a6c9db2e8473270b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 2, 24, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b458d27745ffd7da6fa1d1c33d215cab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([11, 24, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_96bf7284eaf0792ab456d9db96a50e86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c93a70902343a6481859d47a82215837
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 64, 150], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b4d58ae2cc15dc894777188ea4a1f5f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 3136], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ec2e99195e13b53f7a60e717fbf2b03d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 136, 160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_27e0afb8fe9b0617c63f0bcdfbee8187(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 68, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bee6c812ec3a0b7a8c554fe718601220(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 34, 40], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_888f3ccc82fa9ddcadcddb68450d9741(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 17, 20], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bd536d1f3e303e644f0d3960fc76230(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 9, 10], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e151c919078af8431df1c59ed38d43f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 136, 160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e4c89e6ab66eb99d9c197af0ae3ea533(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 68, 80], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7ea672eb50437d7a843853b9678445b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 34, 40], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b9f46dc2124add3ec46964525255c423(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 17, 20], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4694fe364bc2ec756810335a97a99d92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 9, 10], dtype='float32', min=0, max=0.5),
        ]


class PrimitiveOp_022fd0d0fbb91d8d610f4f9a20d16d0f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle.transpose(input_0, perm=[0, 3, 1, 4, 2, 5])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e9aeb3671daf3da02483c54f5ae7db79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_022fd0d0fbb91d8d610f4f9a20d16d0f
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 16, 512, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_57bc19756be1bf406f5912adf446b150(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_377f510e75bdb2acb427eab9c9051a55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_12cb9eeeaba75c523c07c3f282bee05b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f77bcdc90d2aabbf93e8b893de2b814e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99e35bc82e24367d51e280f0853a2834(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6ec22d839bd14b973198483a92cee7fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_022fd0d0fbb91d8d610f4f9a20d16d0f
    def get_inputs(self):
        return [
            paddle.uniform([1, 13, 13, 512, 8, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e8ea21ea31c94482eee0b12342c555e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_75907173815c6a8f2f70c6fe28b21555(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 3136], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c39dd5a45dd2467b04d7b720ddff078a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 5, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d569da12e6b7541e24cbb2799e7882be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04e74dd5e738257bc892eb87deec4a41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9263d4120fcd5a36ea1ff1d6fc99b397(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 2, 5, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1e304f4f5de7388cbee3e90ad42803d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 512, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_300b60a69d34f9db9e1b9c595b073f8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 2048, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b730e123118828e52b8914f8c9b244d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 8, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_421b4d40c7b601dab0cc492730fb3e01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 2, 8, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2442f77297b4c79f87fe8f4517bb8ba1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c06279daafa1419438bf752e26b80dc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83274236be6d9052af12fd6b65be34d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 6400], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d68410b0ce1a5440fd33e792d6167fec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 6400], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1c603c4619fbe5b17175a741a5a19e1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 3600], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a19fcd1570e929d09c2d11b2aa6c71b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 3600], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5bc89cb6efce184e32dd669b412d5a10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c93a70902343a6481859d47a82215837
    def get_inputs(self):
        return [
            paddle.uniform([16, 32, 64, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6b1e16182e3b4bbc61653da958fa8ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([16, 64, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_70082c5af70d602bf5e30551d30500ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([10, 200, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6a8a033b7ff63f2155e07822340f533e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 100], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5c0231f5135d9d82553cda0bc0e5534a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 6, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_938450b0d47ac7a4342ef4aeff19345a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c565d4dd07791abcaab8b0fc2655359(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 49], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e61e00430f37ed8a99145862bf94e638(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 2, 6, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ac61136039a5b7e5e55e87930b435ce8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([11, 6, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4aaa7636d4fc4c54a6a789e4c4a30be2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 3136], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_052ffacaa912c24b5ca17cd50a67e1a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 3136], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_275ce278f3ac1049bad32afd63d41076(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 9216], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1a9bac091a42b42c8fa8955ef6ff694c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 9216], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_df3f9dee7fd995dcc58380a7b71becf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 9216], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_08b1ebc52bdff1c3ef0b3564c7912882(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 2704], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_75e24719522cf8f1ed284e18e644ad03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 76, 2704], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8f2feb6c684f5add2413bdb906394f93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6923c7c04758b9b93d470c599b9855c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 232, 16, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e29e9d891003950644688c23ffe47698(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 7, 1, 7, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_522a491dbe8b7a562a6bc0bd7ebaa1a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d86c553350cb977b271d6b77a7c3701
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 49, 3, 24, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_419bfe7d76953289487325884f1a7bae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c81b4cc87754d47601950d8e7ad403bc
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4c4fae44f61dade18457dbac98ddb75f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([54, 197, 3, 3, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0e4f8a917d5b895f6db4d010b63cf3e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 197, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4052f2c78d7d5cadc45eda547235a246(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([54, 3, 197, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_75c9bb15a71ccc50859176566f4d7844(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6923c7c04758b9b93d470c599b9855c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16, 128, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_67886eadb449e49ad175809564cf55c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 32768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cab069b0ee2bc7aac6b8db5f0e6bb04a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 1, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6caba764d13d2758a17a4ec2a0bd616a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_948159a28311199105605fbcbe40737a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_899356e965b5d5ce9b6889a028bb8501(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 2, 1, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_88fcf2ffc0eeea7ceff5183862a53650(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 1024, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_722e68a0369cc78e4a9d0dee4fdd58fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 65536, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c1faf881e36c9693a1275cbc7c1a8f02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff3b4a1a7327cf7244031390082a7911
    def get_inputs(self):
        return [
            paddle.uniform([300, 256, 49], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a6d508b858c41bb25de5355f969d327a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([10, 640, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8660edb0ad6af72253f03f79b8bdfbd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 320], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_12cb9eeeaba75c523c07c3f282bee05b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_83274236be6d9052af12fd6b65be34d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 6400], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ab23f48d91cf020396848f2df4dbb720(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 6400], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8c46772c39f4005cc226f4c2ea04ab77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 6400], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_20ab57c76260eae484b7121ef1d55fd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 169], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_246557cd47082a376e0a70a5dffff7fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 169], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a3ff431456803c712feea64f3fb7178a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 768, 49], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bd4d23f636ae6a537492c2bd14ea8d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 676], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8ec46505c18ea8dc84a44773eb6fc7de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 676], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc724c10cd08c1f0a15b73b5b0679765(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 529], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_107bfcd0a3adf42147cdac125ac13d7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 529], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5fafb6f77a6ade405b2096da7284c66e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 7, 4, 7, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c74a72ca485e66200fd8b5feac3246d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d86c553350cb977b271d6b77a7c3701
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 49, 3, 6, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3fc662472882b37a976cf357cd9c4015(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c81b4cc87754d47601950d8e7ad403bc
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 6, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_803b89292a95d11cc342725eb722640e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a86201b8325869840fca2f0b10cb1137(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_efbcd7d82476d7bd965eaa2c1b1f9977(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b4e2e5bd47b0fa3e5d97bea40fc1c4c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c93a70902343a6481859d47a82215837
    def get_inputs(self):
        return [
            paddle.uniform([8, 16, 32, 160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7d1e8d570f4eacf4253755ae95e62ec2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([8, 256, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4dd6177979d172d922b0b26c17d281b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([8, 8, 288, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fdfecd612be1260026db37ed9db68506(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([4, 1, 2, 24, 12, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_835501c42ba1b3258f435264e1ab639b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 16384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc3be67777734e4973c0f37908aa6765(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 16384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_79a5522489b82171d22f262f5cd4e078(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 16384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_835501c42ba1b3258f435264e1ab639b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 16384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc3be67777734e4973c0f37908aa6765(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 16384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_79a5522489b82171d22f262f5cd4e078(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 16384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e57e7e8d501ca280d45030b96f4b9464(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1600], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_69c1d9be914407730b434d3341544944(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1600], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_236faf0941775a86ffb222086d4640e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 1600], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_aec3d88a50334d052fefc648b8727778(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6923c7c04758b9b93d470c599b9855c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 72, 14, 25], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f2d1f304f0d3b3da5491482fb2b14c96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 3, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e8ea21ea31c94482eee0b12342c555e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b2e2b169b94aeaca64d03fc79fb84663(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 49], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6f611c801d69db458480d96e35ef5f3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 2, 3, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c508cf6a806c60b3f7ab0aaa0de3d568(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([11, 3, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c6de3d33e41e1a63a721541715d6402a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d0630210f264e4585b44324d4fc6560(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 196], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8f20ca636b29f2226414790c95a24ac5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea3c04f43d1994a7ab54cbf907ac684e
    def get_inputs(self):
        return [
            paddle.uniform([4, 8, 8, 128, 4, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5e1f651613723189065e0030ff8eb5e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7b72744bd6ba1ae88e5a0659331d17d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([10, 96, 40], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f8ab3ad124a49d2636cc5c9e3d2516b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([10, 320, 3, 4, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a3e3e3ba86e2730e2cbc13f19e056b6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a8591be6adcedce1b6b1c841be77eb7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_794344516d8987b79da0ed905a5c19e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 361], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_27ff35a740674260ccd2b2675b5b4574(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 361], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_12cb9eeeaba75c523c07c3f282bee05b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f77bcdc90d2aabbf93e8b893de2b814e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_99e35bc82e24367d51e280f0853a2834(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9f38845d0b8d10f9ce93dd531397cf80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 1, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0a63e409d1ebf7a98332618a580eaf49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_96c4fe09cdc2856b5dd7259cb77adee6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6673d15e04b6905172f6a3c323d60d86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 2, 1, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_66615c76c960e29f45e1157f79d7c204(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 512, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_75ae15f4f9381a3e3fb0d3ea7375d7d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32768, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_491226e6b86b40afae5f2b41a9d045b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 24, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_82889a7f391abf28a6c9db2e8473270b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 2, 24, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b458d27745ffd7da6fa1d1c33d215cab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([11, 24, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_96bf7284eaf0792ab456d9db96a50e86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c93a70902343a6481859d47a82215837
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 64, 150], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fcece4f594476356b9a581dd2bd179eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f77bcdc90d2aabbf93e8b893de2b814e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f60bb085e7f4a115d7ef735a4a1db4ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c6f4c69b2cca86cf1e7f6c4e914e69f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([10, 200, 3, 2, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b08a257022d1543aeab946cf0383098a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 200, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6b69c51e4ec4bb28f57d818dc9f7aa22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 200, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d299fba10aff5047244cc1addb996227(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c93a70902343a6481859d47a82215837
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f3389e83558ca3b52d31ebf57b1f0e97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_96512d367662fe2593a37b039404d3c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1006b823370d28fe0b4e283c48eeec6b
    def get_inputs(self):
        return [
            paddle.uniform([16, 49], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_90af46988b759ef34bc68b49879639de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1006b823370d28fe0b4e283c48eeec6b
    def get_inputs(self):
        return [
            paddle.uniform([784, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_623c1c86b3f82ea714e86d63cf68ba08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 49, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05780c471d1bae610137fb0b094b207f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3f9be6e000823f472de1247a4eba257a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([22, 197, 2, 6, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e38a71cc726d3817612464fe2921bcd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([22, 197, 6, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eda53cf5bbbea12cc977028f97747e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([22, 6, 197, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3da1e831fc91e217cebc36140f1bd25c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_320152e719051e3b6f5ddf77d540888d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 50], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_03229ffa13f3a3ea578e2a66bb788843(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 21760], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_edf9c07411c0b38e1cf4baaf06d9307f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 21760, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_03229ffa13f3a3ea578e2a66bb788843(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 21760], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0e431714b6fb6ee6a8bef0af6254ecc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([240, 4, 96, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_57ee92bc8481cc6d2f3180d2edb499ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([10, 1, 24, 48, 2, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c20cba52626a45d89d00f5c0250c3c5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([4, 32, 144, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4fa842883bd0678ef88cf6bff666eead(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([4, 1, 1, 12, 12, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eea0451ef2c7a9b8033e6939f263a197(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 136, 208], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_eb87bd0f6af9b69b1166e8996b752e21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 68, 104], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c20ff8df53d765caef266ff94104f9dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 34, 52], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_287c8a2de1e32fb7471a60d4746d356d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 17, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_277454c4d4852e5bdcefea7d352a534a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 9, 13], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5b777d7546bcbbc994d3d6119a55a125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 136, 208], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_850ee7e2424876b23d2fa8a8eb09db70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 68, 104], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_45ee70b3766d3d66e5903d6e63013a8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 34, 52], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3d0e2a6dbf1ebad0aa1064b366bab977(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 17, 26], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0db693e62c7d1485acbff19391ff9d96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 9, 13], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3bd4d23f636ae6a537492c2bd14ea8d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 676], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8ec46505c18ea8dc84a44773eb6fc7de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 676], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e3be8a93bd387806bc1cfda8577d47eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([43, 2, 7, 2, 7, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8dac65841b4e352f7fb654dbe1535d15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d86c553350cb977b271d6b77a7c3701
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 49, 3, 12, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0cfda5b14432d5c6bd13c038f10871e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c81b4cc87754d47601950d8e7ad403bc
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_17ba62a7ccd8b492c9a20b7bfb744090(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 4624], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c06b530a443f7100bb668607ea1b6e86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4624], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_35e4f9dfd76081383df0e06e899e2a84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4624], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5747857babf2aebc051fbde6fd0948e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 2, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_06d98bee067822a32c9ed6606ef8a43b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_07f29c75b49b9e7c3c39a004059f7151(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bbb6cd116054c79ad087d7b8fb77bba1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 2, 2, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3a2b94d832356163a065283367839afb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 512, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_caf3b73498e9677d92a480a4b579d34d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8192, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2655b6d4b710273d82b42f2a4ae1e975(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 5, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0ae927be4e9ce3b0f1ca34d037a70da7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 320], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a36e2242262ca2e94d040e52df16083(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ac0d57ff936015c8ff2acc9519c74893(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 2, 5, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f2ef093316c1c94d018f91c35b646401(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 512, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ce94cb2b1f4e611e63139d13f1b5496f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 2048, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_531a0a4d61cd25f3dd1ea17bb32edfba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 1600], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_95a0c199180bad02ff5a2efc8c924831(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 1600], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_198ac02f444440f7e2bd4976d08ad510(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 5184], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7d9eddc75b54ea13938c0eb42412e2a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 5184], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a8b9b0b785479c8dfa541aee840383c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 5184], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4a5c4e85c7a8104b5f44937bba167707(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c93a70902343a6481859d47a82215837
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ecb0ac1c7722e372242a628b75cb5887(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 8192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6e7be919b3f459bbae9fcd23236b68fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 8192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9385985aae0f75b53e047abd686e032b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 16384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0e2cbec3ac1b217a5be8adbad9667e25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 12, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4ce5453066ba95761a5dc516a2356b9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 100], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ac16e873b6f4228e28c4a6a113ac92ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 100], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0288947982b32424a1354f9f6c281212(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([11, 1, 7, 1, 7, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_003f5dbc84331f5566d7a606680a2c3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d86c553350cb977b271d6b77a7c3701
    def get_inputs(self):
        return [
            paddle.uniform([11, 1, 49, 3, 24, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_48ee5fd2e2893d01236d04ba16df0773(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c81b4cc87754d47601950d8e7ad403bc
    def get_inputs(self):
        return [
            paddle.uniform([11, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05780c471d1bae610137fb0b094b207f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a3ff431456803c712feea64f3fb7178a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 768, 49], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_898353ee3ccf7fe405d00946034797cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([4, 96, 9216], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9dac67de52a9943ca76a8487fa4cc2c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 400], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1a74bd79542132c504ace3126bed75d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 400], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_909eb5dcae8b7a4031c2cd5787049969(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([11, 2, 7, 2, 7, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_614f35e3375b95c63693d41299d7ac90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d86c553350cb977b271d6b77a7c3701
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 49, 3, 12, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a07b4da75cd1766cb566da8b610246e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c81b4cc87754d47601950d8e7ad403bc
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0288947982b32424a1354f9f6c281212(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([11, 1, 7, 1, 7, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_003f5dbc84331f5566d7a606680a2c3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d86c553350cb977b271d6b77a7c3701
    def get_inputs(self):
        return [
            paddle.uniform([11, 1, 49, 3, 24, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_48ee5fd2e2893d01236d04ba16df0773(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c81b4cc87754d47601950d8e7ad403bc
    def get_inputs(self):
        return [
            paddle.uniform([11, 1, 24, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f05c372f7da8d1532e5abb132e24b553(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([1, 1025, 3, 12, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d219fb3f1af4994a4755fdc5934de076(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1025, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_af5773269757f29ccb4530c83e84fc33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1025, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91fbd7acdd80c3de7a6e36aad42dc0f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 768, 49], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4b65df8eeae56b0d487aa3b87375fab3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([44, 8, 288, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_860f8bee02b510585a06003dcbf9a72d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([22, 1, 2, 24, 12, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_909eb5dcae8b7a4031c2cd5787049969(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([11, 2, 7, 2, 7, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_614f35e3375b95c63693d41299d7ac90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d86c553350cb977b271d6b77a7c3701
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 49, 3, 12, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a07b4da75cd1766cb566da8b610246e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c81b4cc87754d47601950d8e7ad403bc
    def get_inputs(self):
        return [
            paddle.uniform([11, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7911c70783b0f03d8796e4b8ccbdfaab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 576], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7b4163d97d42313db37ba31f3a97d02f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 576], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_55291e7f140dcfd38570c66c3bc8848e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e330544749c9c3dbd33751b208084bd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6923c7c04758b9b93d470c599b9855c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 20, 128, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4219fd5257ff3e96ef569965ffa20cf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6923c7c04758b9b93d470c599b9855c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 40, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c8ed1d98e7dcc29d0305c9a1040ebbb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6923c7c04758b9b93d470c599b9855c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 80, 32, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1fd16284b468e44e8d3c5ed105099d4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6923c7c04758b9b93d470c599b9855c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 160, 16, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f186441ad77b02b7cbe4b765910ff683(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 176, 264], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_05492af57aa4e8833a5eb82a20bb3d75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 88, 132], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f7bd59253748cee288637ef694b0824b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 66], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f3ae66c2fdfc38a4b7692da4309f9fb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 33], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f8d73831f14e9b0ad9625ffca3c1a438(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fa9e492d5211eabeb361d9726a7355b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 176, 264], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_912d1a5171132cad24e4c0663de790fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 88, 132], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_589e4c466c6c66857a03e4d10855f8a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 44, 66], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cedef72ce0c30989c92a84c2678e8538(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 22, 33], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_86a20fda0de8fde4fda2888ed8b48e3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 11, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6100e81808992d50e78e9c7ecbba9aef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff3b4a1a7327cf7244031390082a7911
    def get_inputs(self):
        return [
            paddle.uniform([100, 256, 49], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2f58c6482279a76af3b0a6801d0fb441(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 8192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_323bba8b922512a6cf48d51648fd28bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 196], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_774a8ce9a8d0fe1d395e318553738aea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 8, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_35aaf9ed4dde96ed9f7001d7b13e8399(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 2, 8, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e8c424059f4cac462586bd113521abef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_43750ba8d43946ca6ff8900fe9dac38a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1024, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_16b71e4eef5c3d1c4e535e4ee69b2123(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c93a70902343a6481859d47a82215837
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9c93838738d534d816cae3c7d09336e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([11, 8, 7, 8, 7, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_38e78d3f8ab86c8931b19c3233330808(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d86c553350cb977b271d6b77a7c3701
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 49, 3, 3, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_281ce7c83768d01e0f68771c7469a448(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c81b4cc87754d47601950d8e7ad403bc
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 3, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b23e6242f7ef000b6d418cf422c5e4bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 1280], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_283d841286981996dd0824126ebc7921(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1280, 2048], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cb2f6487f7b90225bad60cb36ecc97a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91fbd7acdd80c3de7a6e36aad42dc0f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 768, 49], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e3be8a93bd387806bc1cfda8577d47eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([43, 2, 7, 2, 7, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8dac65841b4e352f7fb654dbe1535d15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d86c553350cb977b271d6b77a7c3701
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 49, 3, 12, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0cfda5b14432d5c6bd13c038f10871e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c81b4cc87754d47601950d8e7ad403bc
    def get_inputs(self):
        return [
            paddle.uniform([43, 4, 12, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_08b1ebc52bdff1c3ef0b3564c7912882(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 2704], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1ab916d93574d6adc9d168a18fe371e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 2704], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_835501c42ba1b3258f435264e1ab639b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 16384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_29cb9d917e136bd85ad38c3c3b1ce402(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 11, 7, 7, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3b4dfb9cb234b7e4821277621dfe65d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 24, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_85c4daa45134660309045eb5b2a5cd63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 2, 24, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5a734ce90e791704b1d6b58824f00f0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([43, 24, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_067718b7cdbb9833db414acfb11783a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 25], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2febe4f018691c0830927e5ab69927e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 400], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_02b73f239d55a5bb3ff9740c824f2c46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 400], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_58a9c54a5919c7af31d57502b8b5e71c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 400], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3c6527143acc5071be78d88e775e067b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8464], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d9aba10ebded02d5cd85c323c1537ac1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 8464], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_09163381bece63048a62270951ef8b31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([144, 4, 96, 24], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7f65b3b94eee414ce28cf580ea79a21b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98a49f2abdb72bcd4d7efc4d126f7cab
    def get_inputs(self):
        return [
            paddle.uniform([6, 1, 24, 48, 2, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b5881a3a72eb5ef526581f305151516a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 5, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_55ae861da75e970dae94955f828ff60f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 320], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9f5af359c0f26eec6a2e76036660e6cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f71b0c95cb3dbf5858943a8f5925d29c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 2, 5, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8fc54d85744e00c5c1e920de17c63cf6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 1024, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d6ca9bc691423d3d2cd09d3f0e6ebf47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 4096, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c044d76b0df1d1c7bb96bb3128478b74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 200, 272], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d2fb4a096d5c4934b62d4d32cab883ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 100, 136], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f2aaf43daa2897fb4463d19b12ed4b7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 50, 68], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c27b846c73d7538468776e8e5f7f9657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 25, 34], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_55877ae5a13a89d5c40c9fe62e831fe2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 13, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ecc670501f750c086c8835ff52024bc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 200, 272], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6345e39de280cfcd46c82e1ab3758f3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 100, 136], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8315524e56b394bd988546c25b44ac50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 50, 68], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b4291ddaec14742e5ef338d416270c03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 25, 34], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_03e97ebd37ff751f4b6713eeefee520d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 13, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_154bda9c9b62cd6fde0b6794c1b0168c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 5, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_28d69e78a6d9b2a263531d5e4bb6d360(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4096, 160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fe36e2c7f95fbf3156e86b60157bb58b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cf40d674487d1470361001b21ae66e41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 2, 5, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_027e7621f356e682a764a1ef626097c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 1024, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5ff9f189c1cc1ecba6c9d81eaedbb8d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 4096, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_835501c42ba1b3258f435264e1ab639b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 16384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bc3be67777734e4973c0f37908aa6765(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 16384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_79a5522489b82171d22f262f5cd4e078(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 16384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e330544749c9c3dbd33751b208084bd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6923c7c04758b9b93d470c599b9855c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 20, 128, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4219fd5257ff3e96ef569965ffa20cf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6923c7c04758b9b93d470c599b9855c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 40, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c8ed1d98e7dcc29d0305c9a1040ebbb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6923c7c04758b9b93d470c599b9855c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 80, 32, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6c36f4f856098a45a74670983eb0b5be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 12, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_67c83f0b2e82337e5045c38f0b02e762(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_89e3e90d5a8be2d10c5ec0044db458a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 384, 49], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d014172ebecc5e740716c6d0358be713(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 2, 12, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_58bfd1740e7798cca15c652ac2d1e5ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([43, 12, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0c7bf76c2b05fd5eca4c360fc50e73fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 8, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d8ab4ef5600c8e35216ef1bff42797ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 2, 8, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e258ba306e0be67546a9195c2315ff73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e52c15b43ff84e803b0f0c7ecea77522(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 512, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2e58830fb5ea44da69de49a1b493eae3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 176, 176], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_df96b9eea012e895a8bab396e80f03fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 88, 88], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_016d63c5f32b809300fead961dae85b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_11192a8e5569217e06b6e6e432f42899(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9cd78e7a98b218394f46129c16f12dd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c67653a118f04a285cd84c6376f1f7f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 176, 176], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_439595de3053330d246f0ef2abfaf806(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 88, 88], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_007329516f98823489f3810b3d20c19e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 44, 44], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_45bca4415118d3d68c316d3e69b8196b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 22, 22], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2dc28783141bd9afac9fe2b552aa1413(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 11, 11], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_803b89292a95d11cc342725eb722640e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 15, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a86201b8325869840fca2f0b10cb1137(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_efbcd7d82476d7bd965eaa2c1b1f9977(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 91, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_727578ac1fc6a4a165d47da6fc216506(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 6, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5bdd3d31a98dd758df7452f070f5ee7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_641c61f628794dd566e19148b0f1de01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 49], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_edba97be2f842d59217e50ded2eb7b15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([43, 49, 2, 6, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_44d161b0edfd9361767b64bca461c4d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([43, 6, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_45f2587b511e7011b8f86f259483cb05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 784], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d58b86dfc6ad8475a49e8da48a2a0e5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 784], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_60011ad1dbc0e660f360d2e3bc099112(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 784], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d58b86dfc6ad8475a49e8da48a2a0e5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 784], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6d0630210f264e4585b44324d4fc6560(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 196], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_521e4aa49f9d185f3476ea16d815927a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1444], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fe09b544dcfefd06151cb64892cb4f21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 1444], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bb3bcf51d74ae4ca3629d6114c89b1cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6923c7c04758b9b93d470c599b9855c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 116, 32, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_97c20c1a19a62b9fbda514c2e694d53f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 12, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c6de3d33e41e1a63a721541715d6402a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c7db052e1af3bf3bc6e1f7dbb9cea76b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 384, 49], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f021a6a86ea229f3545f42afdb009898(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([11, 49, 2, 12, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_63b7363c9e972009a1984169f699ef42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([11, 12, 49, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_113e5838dffc656774b8842c7041ed81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 784], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3588cca23ed7bf18f644a9d779861874(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1764], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2fccc341a0d9d48fb29cee084a375124(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 1764], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_938450b0d47ac7a4342ef4aeff19345a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_295917a100b27b1c4633e5850dc907df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 784], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d4052de676058ca9d0ef9642556fe50b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ebc101d5cb903a2bff1d11bd881d9833(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f45516f4aec8886784912ce4744a1d4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 16384, 2, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_56257d8b28b79a4c3313538fbfdcd623(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 16384, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e17150670de6c3c80d98d83c2609b8ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bed39815cfeee11f3e26a00d1b95dbb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 2, 2, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c515e905bc4fcb0ea28c826a5946bb02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 1024, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3ef7fa636ad99bd410864c83f5e788ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 16384, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e0e15823d5391e5efd67379c8e666724(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 2, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_49542453581dd3065d59093a9686dc43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 8192, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_96c4fe09cdc2856b5dd7259cb77adee6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_55e02d543e455b3c7b8874cd5fcad121(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 2, 2, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2a23b2e2e60ce1bcae0c76b956f50895(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 512, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4d9d891f3505fbc2268eef3a8c4aa37b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 8192, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7963cd7d2373502d2b719a1253d546e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 184, 280], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4d7f82869ef89de41dbbdf8cf5103c04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_63bc82fcb9b65b99825eca492668ad3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_feb388e763b0bcd6836bb3d7728d0c50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db19d4928b0df942b78919a6834a782b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d50ef6c86412c458dc3018f35e852de9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 184, 280], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2f2e2b22212efb56e98c825faa68b1f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 92, 140], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_96cc0556ebd9675de6b7b076d87595f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 46, 70], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7eadfa865e64838f6d55f3c8023ec76b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 23, 35], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e519be66ccc40821eae23ea3fdfb0ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6420236c9f428c410b72cb960108ba47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c93a70902343a6481859d47a82215837
    def get_inputs(self):
        return [
            paddle.uniform([4, 16, 64, 320], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_df864ea8273819dfa2624e15b71b1a6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([4, 512, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2f91b8cc8ec62fc7f4d1677af7a53a66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 80, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f0ff1c3582fcb3524b7dd1fe2415f4bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([86, 197, 3, 3, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_819eb7f19ded68e489ef034f740a75f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 197, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dad823f02f73538b100dfd89ea12e812(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([86, 3, 197, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1ec4c97ef5335ce272b0e4d6e06fc94a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 1, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_d2b1fefbe9065da4dd801baaa3977811(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 32768, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a8db52f910bee197c425d53c1264d608(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 512], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_051a98aaea449e3ab591219c8004f31b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 2, 1, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5cd64e35c29a9372e111f84385bf66ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 512, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ffe3ccb8ca06c7c8c11ba1f439d4cc68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32768, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_707caaf33fd2748455eaa3cc1660949d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 4096], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_55ce4e6f1aa5bf7552a3e55534ac552e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c93a70902343a6481859d47a82215837
    def get_inputs(self):
        return [
            paddle.uniform([4, 8, 64, 320], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2b394a47749cdde7cc01edcea26e9902(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([4, 512, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b98220e6b328f469468043dc6c13c81d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_f16950b4630d298f075fee05f012548f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c93a70902343a6481859d47a82215837
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 4, 17], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_37b561161da67c46eb1539b937beb134(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([22, 196, 4, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_37b561161da67c46eb1539b937beb134(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([22, 196, 4, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7822643dac89650794ea5292a0e3e812(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([22, 196, 4, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_87baf28bb24366971255b50efffda8a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([22, 4, 196, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cf84ddcc0ecc1554a5366394347f5cf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1006b823370d28fe0b4e283c48eeec6b
    def get_inputs(self):
        return [
            paddle.uniform([4, 196], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_7923f44b32373bb3b11a5702ce8dfc5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1006b823370d28fe0b4e283c48eeec6b
    def get_inputs(self):
        return [
            paddle.uniform([38416, 4], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_5ec662ee071d38ea267beea5758578bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 192, 288], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_cc1082f40bcb0a3a2cae844da7af8ddd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e1cb00cc7cd6f42a8c4abbe1c92f4576(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e946daa17e5a177f835a229beea0834f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_db19d4928b0df942b78919a6834a782b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_79b777a0d0818f8ab09c72da2c12876f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 192, 288], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_dd5c53b5fecd23d84b0bf657b73dd17c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 96, 144], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_8106a60351578d46dba191748e66a8df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 48, 72], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_bfbe072ce81558714460d99c267fcb5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 24, 36], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e519be66ccc40821eae23ea3fdfb0ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe82cab8d58e76c293086a19456b745b
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 12, 18], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2d05b232da09ef9dbe725650ffd5973b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([10, 160, 3, 8, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_34c810af701995f398c87ec4b2ac6ab4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 160, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_deba6613f07deaa52135d674164a0ba5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 160, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4796c4cdb0637a4db7954df317c13d44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([22, 49, 8, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4a3d94a471e6706e0beeff4818afe665(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1006b823370d28fe0b4e283c48eeec6b
    def get_inputs(self):
        return [
            paddle.uniform([8, 196], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9e9a8f36e4a828ff1c25be66c6e7dab7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1006b823370d28fe0b4e283c48eeec6b
    def get_inputs(self):
        return [
            paddle.uniform([9604, 8], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_c645e308dad5549faff268aa3dac8ef2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 196, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a1443ad6452c1c462611397703ef9f47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 12, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a1443ad6452c1c462611397703ef9f47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 12, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e7602b80424b3bc9863090c93a907a41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 12, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_04c2c9b35b0f36833bdb93fde1a8cc96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([22, 12, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_9a79bd03276a860d6cdd7f02bb75c0c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1006b823370d28fe0b4e283c48eeec6b
    def get_inputs(self):
        return [
            paddle.uniform([12, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3d6aca7b711d29fbb8ea64b1bdc07741(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1006b823370d28fe0b4e283c48eeec6b
    def get_inputs(self):
        return [
            paddle.uniform([256, 12], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4d8c90f51da4864164e8df7fad0d65c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 3, 6, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1837c4b6701193ac83eb255193c93a65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1174, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_2efdd5f5f0aed08211691f00d9762bed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1174, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_91d86a529eec655f7b5d7fc88e627b53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 150, 256], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_3e44d01368de2a0ef5e7a36e9ae7cebc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c93a70902343a6481859d47a82215837
    def get_inputs(self):
        return [
            paddle.uniform([4, 16, 32, 160], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_fb73d415d2b7f8b9207adda0f1d6c927(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([4, 256, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_a7f090f9a931ec92c85fd5d0465cdd7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6923c7c04758b9b93d470c599b9855c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 58, 64, 128], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_17ba62a7ccd8b492c9a20b7bfb744090(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 4624], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0bcf175066a84c1f876c0b6384fde365(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 68, 4624], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_1600de5680154a729aee017fea8878c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 2304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0486d3530032ec023f082f454df22492(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 2304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0198b605ee24e8c884b26a8a30e91035(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2304], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_6ee6dd400aa8aece5a3045b95f124375(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([1, 1174, 3, 12, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b25fc5a28e1a233f188577f841309c93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1174, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_0da3b26bcd24b716a9da2ed037fc9503(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 1174, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ed3364f439ad62a9c3a784133a0a8c98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 1, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_672f945f88429dbb24a889a8d5fa8e02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 65536, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e17150670de6c3c80d98d83c2609b8ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1024], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_030f5e5ae497475afe220b90e2eca299(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 2, 1, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b1d26433a56352d0e632aa512acd4ad5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 1024, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_4a7c42e5d15e9f03b88debf4851faecd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 65536, 64], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_75907173815c6a8f2f70c6fe28b21555(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 3136], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ecadc7b47823cae385090df713f327cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9bb0ab8ee22c9eaea17bbf071fb0856
    def get_inputs(self):
        return [
            paddle.uniform([10, 50, 3, 8, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_92fb079f6e54db37ba279a55e9e28492(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02db1d6dfbb0c153d730ecc3809e8442
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 50, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_14506344d258b57393eaf1c3a42e9ac5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 50, 32], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b0f8fe4bba876031b5f6a56970f64115(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_b4d58ae2cc15dc894777188ea4a1f5f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec48a690c1f8119f96b9b20832556439
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 3136], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_e0bb99b60c9dab35edf8c6aabbb1a238(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([22, 49, 16, 16], dtype='float32', min=0, max=0.5),
        ]


class TestPrimitiveOp_ba380155ed90f84b27b2ba389a00aaeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b6322821a334ee41da5fa95fd1a1c90
    def get_inputs(self):
        return [
            paddle.uniform([22, 49, 16, 64], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()