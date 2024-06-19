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



class PrimitiveOp_ed4b92cc994dae57ccc5ef21cd854d95(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3024], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_59381ef6b602138d9d7236d1d880a1f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ed4b92cc994dae57ccc5ef21cd854d95
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_8c040199f6fbda31c2782e1c48c87306(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3024, 4], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_635d106389efa2043423dd539a50098d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c040199f6fbda31c2782e1c48c87306
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 3024, 4], dtype='int32'),
        ]


class PrimitiveOp_3e0d820df191fb74bb4c161cb34c6478(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_42bd6040494583cbf19afa0fed22392e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e0d820df191fb74bb4c161cb34c6478
    def get_inputs(self):
        return [
            paddle.to_tensor(6168, dtype='int32').reshape([]),
        ]


class PrimitiveOp_06d67ea732b58de1bd5e29078495b62d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3024, 1], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_088711616939697256fc4d2840e9d7c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06d67ea732b58de1bd5e29078495b62d
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024, 1], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_ae6cb3d616044ff184f0c31db184b0f6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3024, 68], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bd718145ffe403ba170cab78edf7eba8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae6cb3d616044ff184f0c31db184b0f6
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 3024, 68], dtype='int32'),
        ]


class PrimitiveOp_5112f78ee4869e7379a4712163f5af30(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1542, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4a8b8a11c32e98e46d3aded55540528d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5112f78ee4869e7379a4712163f5af30
    def get_inputs(self):
        return [
            paddle.uniform([1542, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_6f393e00b1b3345a379ab204ae097e84(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1542, 4], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aca49031e065c4f76efb0ec87ac85b29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f393e00b1b3345a379ab204ae097e84
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1542, 4], dtype='int64'),
        ]


class PrimitiveOp_acad1b6d15f9c4153bda3d8a99f420d4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4725], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_acf270fc5529c009271adc2a4fa2a02c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_acad1b6d15f9c4153bda3d8a99f420d4
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_a08b14606d789ec62bb8d8edbc17789b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4725, 4], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6ec425588898acf0ea7d168d60df1e2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a08b14606d789ec62bb8d8edbc17789b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 4725, 4], dtype='int32'),
        ]


class TestPrimitiveOp_049f654e0c502d89a5c965c8eecf15e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e0d820df191fb74bb4c161cb34c6478
    def get_inputs(self):
        return [
            paddle.to_tensor(9444, dtype='int32').reshape([]),
        ]


class PrimitiveOp_661bdedd9362524196ae517f04ae6441(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4725, 1], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_982609a26ae840acce7745e39ec37c42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_661bdedd9362524196ae517f04ae6441
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725, 1], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_4bcadaee065e0ad0a14e20bdbdc12f6d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4725, 68], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f6def4c6b7932f218c73da8e0ad3f4e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bcadaee065e0ad0a14e20bdbdc12f6d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 4725, 68], dtype='int32'),
        ]


class PrimitiveOp_93b0c7b727197f39a90c6d411d3e14f0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2361, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_691e91ff0fa4fd201fb7ba1ca11aa87e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93b0c7b727197f39a90c6d411d3e14f0
    def get_inputs(self):
        return [
            paddle.uniform([2361, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_1dc399590b0827959d05e16922d3d408(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2361, 4], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cec862d677e4d907c6a941edb64810fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1dc399590b0827959d05e16922d3d408
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2361, 4], dtype='int64'),
        ]


class PrimitiveOp_45201a49f5c489e6ae24b500d9b50fc3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_427726f97a88f8f83c5444ddbf7a084e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45201a49f5c489e6ae24b500d9b50fc3
    def get_inputs(self):
        return [
            paddle.to_tensor([100.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_1d396cc72fc1944924f674fdda50abe8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0025bc7ba8dfe04cabaa1dfe69b9ff39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d396cc72fc1944924f674fdda50abe8
    def get_inputs(self):
        return [
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_225cd86d0a147fc4afa8a03fd9394229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(512, dtype='int32').reshape([]),
        ]


class PrimitiveOp_34af7e4c2a6f206b6e81c96ab6c6e196(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 64, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_68e72a6a61cc70a1b0fbabb2584034b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34af7e4c2a6f206b6e81c96ab6c6e196
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a7849b832fe73200366b2cfeb97f2693(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 1, 1], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_db86caf7663ec4c8f3efad9c0d923a7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7849b832fe73200366b2cfeb97f2693
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_9478d280232355ab9902d47d54749ccb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_14eefde58eb963618fcbac5d89d71f3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9478d280232355ab9902d47d54749ccb
    def get_inputs(self):
        return [
            paddle.to_tensor([False, False, False, True, False, False], dtype='bool').reshape([6]),
        ]


class TestPrimitiveOp_c6a670db1508309c6013399caa052299(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9478d280232355ab9902d47d54749ccb
    def get_inputs(self):
        return [
            paddle.to_tensor([False, False, False, False, False, False], dtype='bool').reshape([6]),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_7bc8eeffbbc886c484fdfc552b626716(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(256, dtype='int32').reshape([]),
        ]


class PrimitiveOp_b0c435ab25f1af43b55b7413d1671866(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7aff52ef31e5996676e1ae7fe2f4c2c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b0c435ab25f1af43b55b7413d1671866
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_a8c48ab8a91874df17a5c53f5656df6c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 1, 1], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cbfb7085a1453375a8a50e43bf679e47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8c48ab8a91874df17a5c53f5656df6c
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 256, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_0025bc7ba8dfe04cabaa1dfe69b9ff39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d396cc72fc1944924f674fdda50abe8
    def get_inputs(self):
        return [
            paddle.to_tensor([4], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_20e922b4b6e75c1c5cb60e636c8a1e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class PrimitiveOp_b1cd01119c19049abfa5fe3d56771225(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 128, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_be2eecf5a429478c5abd6829b8310056(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1cd01119c19049abfa5fe3d56771225
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_3d2e0160185b57b046b9fbf67bb58c86(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 1, 1], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ef419b58c7e897df83a0c99ef8d0220d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d2e0160185b57b046b9fbf67bb58c86
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 64, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_20e922b4b6e75c1c5cb60e636c8a1e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(64, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_be2eecf5a429478c5abd6829b8310056(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1cd01119c19049abfa5fe3d56771225
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 256], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_ef419b58c7e897df83a0c99ef8d0220d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d2e0160185b57b046b9fbf67bb58c86
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 64, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_9ebf0cf29337c314ccf94b913a219ae5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(192, dtype='int32').reshape([]),
        ]


class PrimitiveOp_30407e04dde1abc0b116b52e6babccc6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_34e14d0c7f23199f0efd64f1cba78a01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30407e04dde1abc0b116b52e6babccc6
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_de562efec81f78f5dbec1af75c650af8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 1, 1], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e2975575703795392fc93df970f61c08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de562efec81f78f5dbec1af75c650af8
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 192, 1, 1], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_503a0a713d33c2562f6d124c48f21506(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_01b0877d3c004cd5cb588d0d70a1bf73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_503a0a713d33c2562f6d124c48f21506
    def get_inputs(self):
        return [
            paddle.to_tensor([0.30443012714385986, 0.18485862016677856, 0.27507275342941284, -0.2047252357006073], dtype='float32').reshape([4]),
        ]


class PrimitiveOp_a332621205b828c0d4f79b40a53ad5fc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_db0d057bf3de8df5813f561a853da3d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a332621205b828c0d4f79b40a53ad5fc
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1, 1, 1], dtype='int64').reshape([4]),
        ]


class TestPrimitiveOp_6d11059e78a4a495e65a37fe1a1fb772(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a332621205b828c0d4f79b40a53ad5fc
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0], dtype='int64').reshape([4]),
        ]


class PrimitiveOp_799b0b481eb4c0091019454328b2472e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[152], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5e1ec46b2afb8d14c458d37a8d83cbc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_799b0b481eb4c0091019454328b2472e
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[152], dtype='int64'),
        ]


class PrimitiveOp_e255e6c23bc992966d1077b792872ffe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_854e644b5e99a7c76018f8c6734741a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e255e6c23bc992966d1077b792872ffe
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[100], dtype='int64'),
        ]


class PrimitiveOp_7d4d294cc65230dc31138897576889e2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, 152, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2b3813f35323328442bfe14efe128877(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7d4d294cc65230dc31138897576889e2
    def get_inputs(self):
        return [
            paddle.uniform([100, 152, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_43d2434285ebdf31b3024c71f6c9bb30(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, 152, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_cd5a49f210ce6677530eb302e8ca6392(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43d2434285ebdf31b3024c71f6c9bb30
    def get_inputs(self):
        return [
            paddle.uniform([100, 152, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_8ebbb6f7a10d61bf9d06173a9e0b390d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[76], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_415163b87d67ad4f5cd1baac60d5393a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ebbb6f7a10d61bf9d06173a9e0b390d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[76], dtype='int64'),
        ]


class PrimitiveOp_2de07cf84782a2c250cadea55b33c3e8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[50], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a46181811b8012b7dd41774696063d3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de07cf84782a2c250cadea55b33c3e8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[50], dtype='int64'),
        ]


class PrimitiveOp_4d8185a205fd4fcb2709b9475ef0dba5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[50, 76, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5cad493bf421cf50afb32b696f748663(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d8185a205fd4fcb2709b9475ef0dba5
    def get_inputs(self):
        return [
            paddle.uniform([50, 76, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_16a07c82005c2fe4768234e5f8e5ff96(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[50, 76, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_18626ea1c73e693f7f9a04995b1d3de0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_16a07c82005c2fe4768234e5f8e5ff96
    def get_inputs(self):
        return [
            paddle.uniform([50, 76, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_fb9118d1fabc8c15e0c2ff56d9046a64(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[38], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1e6cd7a832d00a5ce593c16bac63ec52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb9118d1fabc8c15e0c2ff56d9046a64
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[38], dtype='int64'),
        ]


class PrimitiveOp_02ca4948404d699f3ab456a1ac020480(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[25], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_92142c1c7b5bdfb52fc3770c444fead4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02ca4948404d699f3ab456a1ac020480
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], dtype='int64').reshape([25]),
        ]


class PrimitiveOp_068f821a63f0db56f5e6d37a6b0527e0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[25, 38, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d1f4c5168680436a2cfc7c3673f4cc80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_068f821a63f0db56f5e6d37a6b0527e0
    def get_inputs(self):
        return [
            paddle.uniform([25, 38, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_3ee7bd93bccb1d8c6ddd320b4f6023d0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[25, 38, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7a828d8949191bcbc66f157156d9f20d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ee7bd93bccb1d8c6ddd320b4f6023d0
    def get_inputs(self):
        return [
            paddle.uniform([25, 38, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_80990823450ceb2d17c0920c6454ec93(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[19], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5f77e749e5c3287c22ac205e547578d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80990823450ceb2d17c0920c6454ec93
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], dtype='int64').reshape([19]),
        ]


class PrimitiveOp_a3cb29218907de5b58b704467f95018e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[13], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_22562f47e74f06c0b277a1fa9474c13d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3cb29218907de5b58b704467f95018e
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype='int64').reshape([13]),
        ]


class PrimitiveOp_19f8b21573be9decf6a8a665c454b4c1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[13, 19, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_33cbde48803e9168f2fb09d6604f405f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19f8b21573be9decf6a8a665c454b4c1
    def get_inputs(self):
        return [
            paddle.uniform([13, 19, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_bd63c9a58f931c1d4b2aacd64f9f7c3c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[13, 19, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7cea9168be4b314d52010bb38543237d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd63c9a58f931c1d4b2aacd64f9f7c3c
    def get_inputs(self):
        return [
            paddle.uniform([13, 19, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_9f0e60e72f075637eb4c0bf463bf4d32(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_56965883fd0fad1cc68f1a59723a0588(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f0e60e72f075637eb4c0bf463bf4d32
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='int64').reshape([10]),
        ]


class PrimitiveOp_b7a01867df85272a9f8d629403e1c948(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[7], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_af27e2c9e8e8b95e43bfcda6db0b684f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7a01867df85272a9f8d629403e1c948
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6], dtype='int64').reshape([7]),
        ]


class PrimitiveOp_d12de5600cf7096b5d99c04f1590e4e9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[7, 10, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fe13507380cda61a679621be8fe16882(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d12de5600cf7096b5d99c04f1590e4e9
    def get_inputs(self):
        return [
            paddle.uniform([7, 10, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_eba2d2e57834654a378d577af55d375f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[7, 10, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_aa20162f9f0d3ed3567d6fc2461e23fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eba2d2e57834654a378d577af55d375f
    def get_inputs(self):
        return [
            paddle.uniform([7, 10, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_43de1cc3044190280f1a42cb896a7636(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d396cc72fc1944924f674fdda50abe8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_6414e9cb5aeab69878624f5546e844fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45201a49f5c489e6ae24b500d9b50fc3
    def get_inputs(self):
        return [
            paddle.to_tensor([300.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_6414e9cb5aeab69878624f5546e844fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45201a49f5c489e6ae24b500d9b50fc3
    def get_inputs(self):
        return [
            paddle.to_tensor([300.0], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_cd6aaeb750a50459ec34ce64f49b4f1f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4116], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7b4eb72551b5955733884a929b816e76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd6aaeb750a50459ec34ce64f49b4f1f
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_33988b180b9c91806a7a63416c23173d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4116, 4], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_affef467abc22ccea3ed0440f400d7c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_33988b180b9c91806a7a63416c23173d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'),
        ]


class TestPrimitiveOp_370051da9997c07dd322e6f011ae5aa8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e0d820df191fb74bb4c161cb34c6478
    def get_inputs(self):
        return [
            paddle.to_tensor(8212, dtype='int32').reshape([]),
        ]


class PrimitiveOp_f797db38af0652bbfba2f8521b072451(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4116, 1], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1415c717465b40aed3886713d522bb96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f797db38af0652bbfba2f8521b072451
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 1], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_8d26a657649ea0594e9bdf80b6fbeb27(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4116, 68], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_76749605f8180f30b08a3e8d66e07725(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d26a657649ea0594e9bdf80b6fbeb27
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 4116, 68], dtype='int32'),
        ]


class PrimitiveOp_e02a68af7c8f994fe786ba4dbcdddcbd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2053, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_54a88e3647277e80c205076d270acbbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e02a68af7c8f994fe786ba4dbcdddcbd
    def get_inputs(self):
        return [
            paddle.uniform([2053, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_120a68801710e7e91876e42a1d788d5c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2053, 4], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a0380f14d758089a4e452d9c4be49c45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_120a68801710e7e91876e42a1d788d5c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2053, 4], dtype='int64'),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_c6309e4495c24ad71d476ccb6410e242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class PrimitiveOp_623d28febb5adf1faa5405d3d56324f1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5ba6e4e51d578dfb5552b11417b83e6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_623d28febb5adf1faa5405d3d56324f1
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_bbbb11b99efb50395ec4043d7a8d6ebb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b261dcf7eb61190729852a3da6f10a42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbbb11b99efb50395ec4043d7a8d6ebb
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 128, 1, 1], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_168739ac94130efbf1698bee42da5ead(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[256], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a239f12699213a419aa0ab36dcba1fee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168739ac94130efbf1698bee42da5ead
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[256], dtype='int32'),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_7bc8eeffbbc886c484fdfc552b626716(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(256, dtype='int32').reshape([]),
        ]


class PrimitiveOp_31c83153cd11d968a37becea3e407770(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 256, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2b3d83d67e7831429b3c6a733a95b44f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31c83153cd11d968a37becea3e407770
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 256, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cbfb7085a1453375a8a50e43bf679e47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8c48ab8a91874df17a5c53f5656df6c
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 256, 1, 1], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_6f83ead9f1391547f428b411ea7cc4e5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[80], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7e19f72f995946310116d8ae6d42f149(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f83ead9f1391547f428b411ea7cc4e5
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[80], dtype='int64'),
        ]


class PrimitiveOp_0b0c73d4c256c4255c2b4fb65f43dd95(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[40], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_261f5b6ac5d6a015816bc60f72e75fa8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b0c73d4c256c4255c2b4fb65f43dd95
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[40], dtype='int64'),
        ]


class PrimitiveOp_3b21bfbfb24128ef45526e251e7c4e58(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bd9f764bc3db097b9808ed2ee2a0b250(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b21bfbfb24128ef45526e251e7c4e58
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype='int64').reshape([20]),
        ]


class PrimitiveOp_2d6befd9b4a95789b5118b1d0700bf47(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8400, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_963346bcc2e0beb2d8552110cac5e074(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d6befd9b4a95789b5118b1d0700bf47
    def get_inputs(self):
        return [
            paddle.uniform([8400, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_963346bcc2e0beb2d8552110cac5e074(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d6befd9b4a95789b5118b1d0700bf47
    def get_inputs(self):
        return [
            paddle.uniform([8400, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_74803e472818df185c306091156cbd95(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c3aac30dc5cefb6592a8e5f28070ce0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74803e472818df185c306091156cbd95
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4355631470680237, 0.465285062789917, -0.430935263633728, -0.3135901987552643, 0.17000901699066162, 0.14725714921951294, -0.48706769943237305, -0.4442668557167053, 0.15751606225967407, -0.1798284351825714, -0.3336707353591919, 0.42167603969573975, 0.17849797010421753, 0.0706479549407959, -0.47450143098831177, -0.12924033403396606, -0.4698745906352997, 0.48139137029647827, 0.45962780714035034, 0.15411460399627686], dtype='float32').reshape([20]),
        ]


class TestPrimitiveOp_7914332f4b5eb1edfcd56550f9c7090d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b21bfbfb24128ef45526e251e7c4e58
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([20]),
        ]


class TestPrimitiveOp_d724c72e68cc04d1751fc0096853e057(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b21bfbfb24128ef45526e251e7c4e58
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([20]),
        ]


class PrimitiveOp_635e2719562390e0d417b33ab5360fe9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3549], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e661682815b22d01c35aec758ed4524a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635e2719562390e0d417b33ab5360fe9
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_65e472f51c910f4abc170a023dcbc990(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3549, 4], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7c7510712221e614efa19c73594bcf11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65e472f51c910f4abc170a023dcbc990
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'),
        ]


class TestPrimitiveOp_8d189e150afc42413097e0fc3dbdd6b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e0d820df191fb74bb4c161cb34c6478
    def get_inputs(self):
        return [
            paddle.to_tensor(7300, dtype='int32').reshape([]),
        ]


class PrimitiveOp_de118096f4c00b7740edd95b5ae46aba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3549, 1], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6117c3dbefc1b4f93a60c900dcbab694(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de118096f4c00b7740edd95b5ae46aba
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 1], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_5d91d46da38a6aafdb1fb71922d92f0c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3549, 76], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c165a678b136c0e8d158afaddc05a710(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d91d46da38a6aafdb1fb71922d92f0c
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 3549, 76], dtype='int32'),
        ]


class PrimitiveOp_fd7c2321a05dee5203808580bca24eab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1825, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_427f053736db8c0d2d55f10e1da81faf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd7c2321a05dee5203808580bca24eab
    def get_inputs(self):
        return [
            paddle.uniform([1825, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_07fa0a4501be8020819b2e518a853c0a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1825, 4], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_31241845d57ff6491f353f246d07a68e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07fa0a4501be8020819b2e518a853c0a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1825, 4], dtype='int64'),
        ]


class PrimitiveOp_0bfe0a1ca9e9495647a63b4e675a11ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e299d4e2ce53d8b01faee57c0cca1b8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0bfe0a1ca9e9495647a63b4e675a11ef
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.48465049266815186, -0.04762253165245056, 0.23276299238204956, -0.06612768769264221, -0.49176597595214844, -0.06729069352149963, 0.3730962872505188, -0.010074377059936523, 0.39248335361480713, -0.21158581972122192, 0.41349464654922485, 0.32767152786254883, 0.4794207811355591, 0.046976566314697266, 0.16938143968582153, 0.22565364837646484], dtype='float32').reshape([16]),
        ]


class PrimitiveOp_4bfbb0d4cd6c81b6f8cfc01ec7a6ea50(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f1481a3b5f2a16c80862833cc5f7015c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bfbb0d4cd6c81b6f8cfc01ec7a6ea50
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([16]),
        ]


class TestPrimitiveOp_dcf5d1de461d618da1188251c84d8c4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bfbb0d4cd6c81b6f8cfc01ec7a6ea50
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([16]),
        ]


class PrimitiveOp_fa66f160a3e34528e0d691b1ea16f430(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3a6ab097d020380f9f8b603cef881075(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fa66f160a3e34528e0d691b1ea16f430
    def get_inputs(self):
        return [
            paddle.to_tensor([0.04567500948905945], dtype='float32').reshape([1]),
        ]


class PrimitiveOp_c0c339307df077ee29ceadc7a67febc0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[14], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_61cabe75e54f2ca0b969456f476c86aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0c339307df077ee29ceadc7a67febc0
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], dtype='int64').reshape([14]),
        ]


class PrimitiveOp_4dcfd9c76b3131d7e69112c43e97f12f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[14, 14, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f38c127d5261c66a114407cc72899a7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4dcfd9c76b3131d7e69112c43e97f12f
    def get_inputs(self):
        return [
            paddle.uniform([14, 14, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_968db2a10050de795f41133dbea6ede4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[14, 14, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5008ef2ac3baea9cd426fe557c088d2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_968db2a10050de795f41133dbea6ede4
    def get_inputs(self):
        return [
            paddle.uniform([14, 14, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_3d6ab6bf4351f6379abbd0bc87da57da(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[28], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3f497de691327ccec97cac470cfe6117(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d6ab6bf4351f6379abbd0bc87da57da
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27], dtype='int64').reshape([28]),
        ]


class PrimitiveOp_452e1056fb286a27a6929f62b56a2e98(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[28, 28, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bbd377efb9d1a9e21de4b0a277f006ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_452e1056fb286a27a6929f62b56a2e98
    def get_inputs(self):
        return [
            paddle.uniform([28, 28, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_2bf9a323b46d688b520f5cd1c6534e13(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[28, 28, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2b1df78a27571aa665a90caf0457554a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bf9a323b46d688b520f5cd1c6534e13
    def get_inputs(self):
        return [
            paddle.uniform([28, 28, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b7c1ac54a35b296964fcc1e017e75296(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[56], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_0082887ab6cc6e94790371366790afde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7c1ac54a35b296964fcc1e017e75296
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[56], dtype='int64'),
        ]


class PrimitiveOp_ee045f2bcaadac07b467208c83b86f68(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[56, 56, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_52440dd23d0152b3a9ea60c92a11d363(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee045f2bcaadac07b467208c83b86f68
    def get_inputs(self):
        return [
            paddle.uniform([56, 56, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_9788c8e80ce7a1c98202ccbe608d036e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[56, 56, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b675323ddaaf03626b8d6fa304b36af9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9788c8e80ce7a1c98202ccbe608d036e
    def get_inputs(self):
        return [
            paddle.uniform([56, 56, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_014953e228d3107afdebf529cab6c848(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6069], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e29b2fa7df34a607d2adf78ac9472e5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_014953e228d3107afdebf529cab6c848
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_b882e21cf0f33be86e6ad9ce968ff51a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6069, 4], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5e5eb80f01c4b548361952b4f638def8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b882e21cf0f33be86e6ad9ce968ff51a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 6069, 4], dtype='int32'),
        ]


class TestPrimitiveOp_5ddac20b863de776c0e1ecf092ade494(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e0d820df191fb74bb4c161cb34c6478
    def get_inputs(self):
        return [
            paddle.to_tensor(12348, dtype='int32').reshape([]),
        ]


class PrimitiveOp_21c9131cdbee3b081705db7e8650e0e1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6069, 1], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a692182b8705c36fa2cdd7fe941f76e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21c9131cdbee3b081705db7e8650e0e1
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069, 1], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_eb2885c59641af627e93a1313280ee5d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 6069, 68], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_006220af73758bcbdadb89a123581c11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb2885c59641af627e93a1313280ee5d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 6069, 68], dtype='int32'),
        ]


class PrimitiveOp_eb0b64f975b60ffbb63ae41918c8e6e1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3087, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a4712a6e71fac076ca95b2886fca4440(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb0b64f975b60ffbb63ae41918c8e6e1
    def get_inputs(self):
        return [
            paddle.uniform([3087, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b124f11a3f037d39b4ca2dd26bbd2c74(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3087, 4], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2b0381827762903ddca5b8243a4a84dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b124f11a3f037d39b4ca2dd26bbd2c74
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[3087, 4], dtype='int64'),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_c6309e4495c24ad71d476ccb6410e242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class PrimitiveOp_b4a387e00a223fbc321d6bdc768811ad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 64, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bb054794bf5cfc3544fd1ce32ab84179(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4a387e00a223fbc321d6bdc768811ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b261dcf7eb61190729852a3da6f10a42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbbb11b99efb50395ec4043d7a8d6ebb
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 128, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_7b4eb72551b5955733884a929b816e76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd6aaeb750a50459ec34ce64f49b4f1f
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_affef467abc22ccea3ed0440f400d7c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_33988b180b9c91806a7a63416c23173d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'),
        ]


class TestPrimitiveOp_4eaa0cf3958ca35e7d743ec6156fbf1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e0d820df191fb74bb4c161cb34c6478
    def get_inputs(self):
        return [
            paddle.to_tensor(8476, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_1415c717465b40aed3886713d522bb96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f797db38af0652bbfba2f8521b072451
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_76749605f8180f30b08a3e8d66e07725(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d26a657649ea0594e9bdf80b6fbeb27
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 4116, 68], dtype='int32'),
        ]


class PrimitiveOp_f8fba04ebb451f585d2f21a5cc547ba3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2119, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_5dbc9a83b6ce7fb3182e0a9a3f5564dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8fba04ebb451f585d2f21a5cc547ba3
    def get_inputs(self):
        return [
            paddle.uniform([2119, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_2cfbee53a395b34a10f2a0765569a8dc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2119, 4], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_3bb66da9b8542e9ecfe240f92a9301ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cfbee53a395b34a10f2a0765569a8dc
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[2119, 4], dtype='int64'),
        ]


class TestPrimitiveOp_27808353ddcc94f4a5023e2499eb03c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d396cc72fc1944924f674fdda50abe8
    def get_inputs(self):
        return [
            paddle.to_tensor([7], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_dd5f211485d6277c4da4778daee93036(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d396cc72fc1944924f674fdda50abe8
    def get_inputs(self):
        return [
            paddle.to_tensor([3], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_7c9a909507418b4175eb6ebef5673493(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[32], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_284695197f4d3e7a5e8e522951f94155(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c9a909507418b4175eb6ebef5673493
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[32], dtype='int64'),
        ]


class PrimitiveOp_c7f3c07315ad2188af2b84daedc71666(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[32, 32, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8a2afa1bd507ee8eb5a28e84dbffdacd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7f3c07315ad2188af2b84daedc71666
    def get_inputs(self):
        return [
            paddle.uniform([32, 32, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_3c5bcd52d2431636790407e23aa65b90(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[64], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9bc354cc19c0d684268006ae3711d35e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c5bcd52d2431636790407e23aa65b90
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[64], dtype='int64'),
        ]


class PrimitiveOp_3ac872dcea72fc3a635f4048ca1d5814(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[64, 64, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_8ae2df6c5cfbdfcf2a5a869753c00fba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ac872dcea72fc3a635f4048ca1d5814
    def get_inputs(self):
        return [
            paddle.uniform([64, 64, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_f9d6fba16f32a92a24677d5ed8c85a51(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[128], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_42551dd7687b8c1635477d0b5d92ecb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9d6fba16f32a92a24677d5ed8c85a51
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[128], dtype='int64'),
        ]


class PrimitiveOp_22bbe85eb172340859cb7cd8253de2ed(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[128, 128, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ea3a008aa191a294d8c3ce05f95653b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22bbe85eb172340859cb7cd8253de2ed
    def get_inputs(self):
        return [
            paddle.uniform([128, 128, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_e0ac596d50708f16508305a772d679a7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[96], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9b7b61e1401144b11d9c5480273124ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0ac596d50708f16508305a772d679a7
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[96], dtype='int64'),
        ]


class PrimitiveOp_ae688e83c69cce732f3f0ab63bbf0637(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[48], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_ae07ebf227f1846108b79b54969af586(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae688e83c69cce732f3f0ab63bbf0637
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[48], dtype='int64'),
        ]


class PrimitiveOp_b96b6cfd491ffd9960c841493858128d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[24], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_7fa612e6d8f774e02950956900dc446d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b96b6cfd491ffd9960c841493858128d
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], dtype='int64').reshape([24]),
        ]


class PrimitiveOp_f821d49a59b02e9cd2f2b524b48a1510(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[12096, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d89859d73b896f30ac62fd422a686d9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f821d49a59b02e9cd2f2b524b48a1510
    def get_inputs(self):
        return [
            paddle.uniform([12096, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_d89859d73b896f30ac62fd422a686d9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f821d49a59b02e9cd2f2b524b48a1510
    def get_inputs(self):
        return [
            paddle.uniform([12096, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_225cd86d0a147fc4afa8a03fd9394229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(512, dtype='int32').reshape([]),
        ]


class PrimitiveOp_0a406bfd330bcdfa79c201703642b76f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_896dc68598373300a5d88b38db9520b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a406bfd330bcdfa79c201703642b76f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_db86caf7663ec4c8f3efad9c0d923a7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7849b832fe73200366b2cfeb97f2693
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_284695197f4d3e7a5e8e522951f94155(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c9a909507418b4175eb6ebef5673493
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[32], dtype='int64'),
        ]


class TestPrimitiveOp_8a2afa1bd507ee8eb5a28e84dbffdacd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7f3c07315ad2188af2b84daedc71666
    def get_inputs(self):
        return [
            paddle.uniform([32, 32, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9bc354cc19c0d684268006ae3711d35e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c5bcd52d2431636790407e23aa65b90
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[64], dtype='int64'),
        ]


class TestPrimitiveOp_8ae2df6c5cfbdfcf2a5a869753c00fba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ac872dcea72fc3a635f4048ca1d5814
    def get_inputs(self):
        return [
            paddle.uniform([64, 64, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_42551dd7687b8c1635477d0b5d92ecb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9d6fba16f32a92a24677d5ed8c85a51
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[128], dtype='int64'),
        ]


class TestPrimitiveOp_ea3a008aa191a294d8c3ce05f95653b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22bbe85eb172340859cb7cd8253de2ed
    def get_inputs(self):
        return [
            paddle.uniform([128, 128, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_225cd86d0a147fc4afa8a03fd9394229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(512, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_68e72a6a61cc70a1b0fbabb2584034b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34af7e4c2a6f206b6e81c96ab6c6e196
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_db86caf7663ec4c8f3efad9c0d923a7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7849b832fe73200366b2cfeb97f2693
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_225cd86d0a147fc4afa8a03fd9394229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(512, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_68e72a6a61cc70a1b0fbabb2584034b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34af7e4c2a6f206b6e81c96ab6c6e196
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_db86caf7663ec4c8f3efad9c0d923a7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7849b832fe73200366b2cfeb97f2693
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_225cd86d0a147fc4afa8a03fd9394229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(512, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_68e72a6a61cc70a1b0fbabb2584034b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34af7e4c2a6f206b6e81c96ab6c6e196
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_db86caf7663ec4c8f3efad9c0d923a7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7849b832fe73200366b2cfeb97f2693
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_c593e8c6b6abeae488761917b767345b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(2048, dtype='int32').reshape([]),
        ]


class PrimitiveOp_523ecb4e70a73deee1b3afde7d7d35c1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2048, 64, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_b1eb10e755bd05f7e1028fa8b7b985b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_523ecb4e70a73deee1b3afde7d7d35c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b1ba589dc31bede712445f5f0914341d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2048, 1, 1], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_330ab7ad6b1ec490a0382ec0749bc444(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1ba589dc31bede712445f5f0914341d
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2048, 1, 1], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_76b84f38a3d8917252bbaf22dc1d3d5d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[68], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_4d9bfb3750a55213e4022135d44810bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76b84f38a3d8917252bbaf22dc1d3d5d
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[68], dtype='int64'),
        ]


class PrimitiveOp_04a7407169a437cb91a8f97b937d48f4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[34], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_113dc3e78621a375ecb765d6fd3f9ed7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04a7407169a437cb91a8f97b937d48f4
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[34], dtype='int64'),
        ]


class PrimitiveOp_d0d5dfaa8b40c4f51bd48928b16cb385(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[17], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_2eea0685801d846081a1b8923d36b46f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d0d5dfaa8b40c4f51bd48928b16cb385
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], dtype='int64').reshape([17]),
        ]


class PrimitiveOp_3fd1cdc55631f9dbe35be20b9881759f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6069, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_9b11481b2b4e2fae35095794ce53ae56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3fd1cdc55631f9dbe35be20b9881759f
    def get_inputs(self):
        return [
            paddle.uniform([6069, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9b11481b2b4e2fae35095794ce53ae56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3fd1cdc55631f9dbe35be20b9881759f
    def get_inputs(self):
        return [
            paddle.uniform([6069, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_225cd86d0a147fc4afa8a03fd9394229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(512, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_896dc68598373300a5d88b38db9520b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a406bfd330bcdfa79c201703642b76f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_db86caf7663ec4c8f3efad9c0d923a7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7849b832fe73200366b2cfeb97f2693
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_225cd86d0a147fc4afa8a03fd9394229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(512, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_896dc68598373300a5d88b38db9520b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a406bfd330bcdfa79c201703642b76f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_db86caf7663ec4c8f3efad9c0d923a7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7849b832fe73200366b2cfeb97f2693
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_225cd86d0a147fc4afa8a03fd9394229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(512, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_896dc68598373300a5d88b38db9520b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a406bfd330bcdfa79c201703642b76f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_db86caf7663ec4c8f3efad9c0d923a7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7849b832fe73200366b2cfeb97f2693
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_c593e8c6b6abeae488761917b767345b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(2048, dtype='int32').reshape([]),
        ]


class PrimitiveOp_7ce3402eb1e1a78703abfa6a22d1ed86(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2048, 64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_27e767ab52d7eb23e47a772b8101a96f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3402eb1e1a78703abfa6a22d1ed86
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 64, 64], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_330ab7ad6b1ec490a0382ec0749bc444(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1ba589dc31bede712445f5f0914341d
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2048, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_27808353ddcc94f4a5023e2499eb03c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d396cc72fc1944924f674fdda50abe8
    def get_inputs(self):
        return [
            paddle.to_tensor([7], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_46aca81dc541bfaf18a7c1a5afd3fe9d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 11109], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f6db2df6309b51cc1dd9f89e2b93d264(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46aca81dc541bfaf18a7c1a5afd3fe9d
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_32576a7b1534741c4f479e46b953491a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 11109, 4], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6eb60f119f7b988a0573f26f5da6a92f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32576a7b1534741c4f479e46b953491a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 11109, 4], dtype='int32'),
        ]


class TestPrimitiveOp_fb6ea86dd0d86d7f4fd3948918c29c3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e0d820df191fb74bb4c161cb34c6478
    def get_inputs(self):
        return [
            paddle.to_tensor(22424, dtype='int32').reshape([]),
        ]


class PrimitiveOp_740efdca4203ff36869a8e6d04ce1731(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 11109, 1], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_74c6d004709bad8d08eb80b48c5c452a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_740efdca4203ff36869a8e6d04ce1731
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109, 1], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_0fc3e75a3abf836cfbe0b7f4ee52ae94(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 11109, 68], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c76ea41bae92bf279ff836a06129d244(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0fc3e75a3abf836cfbe0b7f4ee52ae94
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 11109, 68], dtype='int32'),
        ]


class PrimitiveOp_4dc449d98cd16186d647f1cb1dbf2953(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5606, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_762c38ff9df1e038c10b40c91c8ff56d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4dc449d98cd16186d647f1cb1dbf2953
    def get_inputs(self):
        return [
            paddle.uniform([5606, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_91da1e92b39b76048ec41949acdc8c78(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5606, 4], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_92b8500bc750e09b316312055419d845(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91da1e92b39b76048ec41949acdc8c78
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[5606, 4], dtype='int64'),
        ]


class PrimitiveOp_80dd1577d2ba21d05e76c037cf385c43(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2100], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d1ea3aec9d748f3fafb9f94d0037ec47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80dd1577d2ba21d05e76c037cf385c43
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_3936d8f47391a74848a6f8c60717cc42(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2100, 4], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1a43d2c7c6340e11d042afd33b914fe9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3936d8f47391a74848a6f8c60717cc42
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 2100, 4], dtype='int32'),
        ]


class TestPrimitiveOp_a231cb2e23b90031e82d3666f3c97c6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e0d820df191fb74bb4c161cb34c6478
    def get_inputs(self):
        return [
            paddle.to_tensor(4144, dtype='int32').reshape([]),
        ]


class PrimitiveOp_147782535a6c6234184a8ef44aaeb5cb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2100, 1], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_034723bfa1b05ed2a8dc962b8f15cd06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_147782535a6c6234184a8ef44aaeb5cb
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100, 1], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_7a46795b6cab6ee0090890ee02b933eb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2100, 68], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_64236a27e5589f9963f32d382bb14577(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7a46795b6cab6ee0090890ee02b933eb
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 2100, 68], dtype='int32'),
        ]


class PrimitiveOp_6a9447aaf20490634409c8cb196818e2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1036, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_daa5f1b514baf48d945f7dcdfe9f6762(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a9447aaf20490634409c8cb196818e2
    def get_inputs(self):
        return [
            paddle.uniform([1036, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_c6b76632dff0e487d654b3e402ed8bd2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1036, 4], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_07dbe5bd73306b650bb42a5526084d07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6b76632dff0e487d654b3e402ed8bd2
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1036, 4], dtype='int64'),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_c6309e4495c24ad71d476ccb6410e242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_bb054794bf5cfc3544fd1ce32ab84179(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4a387e00a223fbc321d6bdc768811ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_b261dcf7eb61190729852a3da6f10a42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbbb11b99efb50395ec4043d7a8d6ebb
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 128, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_225cd86d0a147fc4afa8a03fd9394229(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(512, dtype='int32').reshape([]),
        ]


class PrimitiveOp_55b94a50b5ef81c536f28f6d618615fe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 97, 97], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_a6b59548f6bd0940985d7ab8a6cdb1b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55b94a50b5ef81c536f28f6d618615fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 97, 97], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_db86caf7663ec4c8f3efad9c0d923a7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7849b832fe73200366b2cfeb97f2693
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_284695197f4d3e7a5e8e522951f94155(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c9a909507418b4175eb6ebef5673493
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[32], dtype='int64'),
        ]


class TestPrimitiveOp_8a2afa1bd507ee8eb5a28e84dbffdacd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7f3c07315ad2188af2b84daedc71666
    def get_inputs(self):
        return [
            paddle.uniform([32, 32, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9bc354cc19c0d684268006ae3711d35e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c5bcd52d2431636790407e23aa65b90
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[64], dtype='int64'),
        ]


class TestPrimitiveOp_8ae2df6c5cfbdfcf2a5a869753c00fba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ac872dcea72fc3a635f4048ca1d5814
    def get_inputs(self):
        return [
            paddle.uniform([64, 64, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_42551dd7687b8c1635477d0b5d92ecb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9d6fba16f32a92a24677d5ed8c85a51
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[128], dtype='int64'),
        ]


class TestPrimitiveOp_ea3a008aa191a294d8c3ce05f95653b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22bbe85eb172340859cb7cd8253de2ed
    def get_inputs(self):
        return [
            paddle.uniform([128, 128, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_e661682815b22d01c35aec758ed4524a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_635e2719562390e0d417b33ab5360fe9
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_7c7510712221e614efa19c73594bcf11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65e472f51c910f4abc170a023dcbc990
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'),
        ]


class TestPrimitiveOp_4c2547e4e1a15a40caee898a7373b1b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e0d820df191fb74bb4c161cb34c6478
    def get_inputs(self):
        return [
            paddle.to_tensor(7236, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_6117c3dbefc1b4f93a60c900dcbab694(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de118096f4c00b7740edd95b5ae46aba
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 1], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_c4ca4be9963981678e05bacd8451f2e3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3549, 68], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c2e2d8c38552a64dc97f1db02b2317c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4ca4be9963981678e05bacd8451f2e3
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 3549, 68], dtype='int32'),
        ]


class PrimitiveOp_eb9181b5ff52def6164dc97a088c1712(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1809, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c3221dc332e3962c98464717972d61c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb9181b5ff52def6164dc97a088c1712
    def get_inputs(self):
        return [
            paddle.uniform([1809, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_5e70fd59590ef4c6ea729ed303c35de6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1809, 4], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c5110cd7464ceb22eccdb6e8adae5930(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e70fd59590ef4c6ea729ed303c35de6
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1809, 4], dtype='int64'),
        ]


class TestPrimitiveOp_a239f12699213a419aa0ab36dcba1fee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168739ac94130efbf1698bee42da5ead
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[256], dtype='int32'),
        ]


class TestPrimitiveOp_284695197f4d3e7a5e8e522951f94155(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c9a909507418b4175eb6ebef5673493
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[32], dtype='int64'),
        ]


class TestPrimitiveOp_8a2afa1bd507ee8eb5a28e84dbffdacd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7f3c07315ad2188af2b84daedc71666
    def get_inputs(self):
        return [
            paddle.uniform([32, 32, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9bc354cc19c0d684268006ae3711d35e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c5bcd52d2431636790407e23aa65b90
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[64], dtype='int64'),
        ]


class TestPrimitiveOp_8ae2df6c5cfbdfcf2a5a869753c00fba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ac872dcea72fc3a635f4048ca1d5814
    def get_inputs(self):
        return [
            paddle.uniform([64, 64, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_42551dd7687b8c1635477d0b5d92ecb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9d6fba16f32a92a24677d5ed8c85a51
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[128], dtype='int64'),
        ]


class TestPrimitiveOp_ea3a008aa191a294d8c3ce05f95653b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22bbe85eb172340859cb7cd8253de2ed
    def get_inputs(self):
        return [
            paddle.uniform([128, 128, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_fbbf721f9dd90e4d18449733353f118c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d9a607f5a53e7bde90ee1f62c408e99e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fbbf721f9dd90e4d18449733353f118c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.03167968988418579, -0.37636035680770874, -0.21670317649841309, 0.39032089710235596, -0.44819122552871704, 0.4351775646209717, 0.23851841688156128, -0.3821527063846588, -0.28143811225891113, 0.29411834478378296, 0.40033769607543945, -0.4059436023235321, 0.4736446142196655, -0.3660675883293152, 0.04382580518722534, -0.24653342366218567, 0.4353167414665222, 0.2468177080154419, -0.38413405418395996, -0.42135077714920044, 0.36219167709350586, 0.0969361662864685, 0.03515017032623291, 0.04207485914230347], dtype='float32').reshape([24]),
        ]


class TestPrimitiveOp_7ea8f49889676bd664c8c948b148cd8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b96b6cfd491ffd9960c841493858128d
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([24]),
        ]


class TestPrimitiveOp_f06436bb8f4f2d83f7b3911c545af1b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b96b6cfd491ffd9960c841493858128d
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([24]),
        ]


class TestPrimitiveOp_42551dd7687b8c1635477d0b5d92ecb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9d6fba16f32a92a24677d5ed8c85a51
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[128], dtype='int64'),
        ]


class TestPrimitiveOp_ea3a008aa191a294d8c3ce05f95653b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22bbe85eb172340859cb7cd8253de2ed
    def get_inputs(self):
        return [
            paddle.uniform([128, 128, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9bc354cc19c0d684268006ae3711d35e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c5bcd52d2431636790407e23aa65b90
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[64], dtype='int64'),
        ]


class TestPrimitiveOp_8ae2df6c5cfbdfcf2a5a869753c00fba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ac872dcea72fc3a635f4048ca1d5814
    def get_inputs(self):
        return [
            paddle.uniform([64, 64, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_284695197f4d3e7a5e8e522951f94155(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c9a909507418b4175eb6ebef5673493
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[32], dtype='int64'),
        ]


class TestPrimitiveOp_8a2afa1bd507ee8eb5a28e84dbffdacd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7f3c07315ad2188af2b84daedc71666
    def get_inputs(self):
        return [
            paddle.uniform([32, 32, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_66cb74165486cb8db056065a8784ff2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bfbb0d4cd6c81b6f8cfc01ec7a6ea50
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype='int64').reshape([16]),
        ]


class PrimitiveOp_dcf423e8a8c00152d2a4d3f2a3d944c3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16, 16, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e7d545d6800741c667f17c7a56f68a79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dcf423e8a8c00152d2a4d3f2a3d944c3
    def get_inputs(self):
        return [
            paddle.uniform([16, 16, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_5beaef702f0af29738f36df1b2675908(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c1c3701b90e8a7f8d9d783e533ff733e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5beaef702f0af29738f36df1b2675908
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype='int64').reshape([8]),
        ]


class PrimitiveOp_7f710d4879a0b36d6f2c72b81d1b385a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8, 8, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d13ad70ad10501bdf4e610adcc026d0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f710d4879a0b36d6f2c72b81d1b385a
    def get_inputs(self):
        return [
            paddle.uniform([8, 8, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_27808353ddcc94f4a5023e2499eb03c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d396cc72fc1944924f674fdda50abe8
    def get_inputs(self):
        return [
            paddle.to_tensor([7], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_427726f97a88f8f83c5444ddbf7a084e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45201a49f5c489e6ae24b500d9b50fc3
    def get_inputs(self):
        return [
            paddle.to_tensor([100.0], dtype='float32').reshape([1]),
        ]


class TestPrimitiveOp_c98ce3ab1ca1343149618d5302dc606e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d396cc72fc1944924f674fdda50abe8
    def get_inputs(self):
        return [
            paddle.to_tensor([8], dtype='int64').reshape([1]),
        ]


class TestPrimitiveOp_43de1cc3044190280f1a42cb896a7636(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d396cc72fc1944924f674fdda50abe8
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_13ad6d900d5aff5d8b6b9d48b868af01(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8400], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_276c3ad51e3ef26b4dc0f69d90083e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13ad6d900d5aff5d8b6b9d48b868af01
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_1e337b65abc56922f7a44586ba9f5a50(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8400, 4], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_21c2218e6f73a0b136728b7bd3f6db2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e337b65abc56922f7a44586ba9f5a50
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 8400, 4], dtype='int32'),
        ]


class TestPrimitiveOp_0f05524ee3098fd610b21cbd99f2943c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e0d820df191fb74bb4c161cb34c6478
    def get_inputs(self):
        return [
            paddle.to_tensor(16716, dtype='int32').reshape([]),
        ]


class PrimitiveOp_2b006d447fdeccf4598d055e342ec949(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8400, 1], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_de20a575725cd7d49e671f7d7cc81c52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b006d447fdeccf4598d055e342ec949
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400, 1], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_4fb60febf02b154e98728c1e82f69459(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8400, 68], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_32081258c79ca24a9350426a4b6f72f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4fb60febf02b154e98728c1e82f69459
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 8400, 68], dtype='int32'),
        ]


class PrimitiveOp_ce1851e074f61fc6d1c12e2fe473d11e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4179, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_fa7e03a7647130a269b7e5c766282893(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce1851e074f61fc6d1c12e2fe473d11e
    def get_inputs(self):
        return [
            paddle.uniform([4179, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_076913e1a482f611c3d2c3b30f727d56(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4179, 4], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bc7b31b2e8ce6963a3b67e338e15a74a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_076913e1a482f611c3d2c3b30f727d56
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[4179, 4], dtype='int64'),
        ]


class TestPrimitiveOp_18e40c9034972b361c0a456f832a6002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


class TestPrimitiveOp_7bc8eeffbbc886c484fdfc552b626716(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f29b6bc53ea5f01bffcf7d92fccbdb85
    def get_inputs(self):
        return [
            paddle.to_tensor(256, dtype='int32').reshape([]),
        ]


class PrimitiveOp_d34f9eac6599e1e2c2a9181b0d5c2f55(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 256, 64, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_99061c89d231e22145a9f5708c66b673(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d34f9eac6599e1e2c2a9181b0d5c2f55
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 128], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_cbfb7085a1453375a8a50e43bf679e47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8c48ab8a91874df17a5c53f5656df6c
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 256, 1, 1], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_e29404b5e2ffedc9bf1f2cbf8ec65b82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d396cc72fc1944924f674fdda50abe8
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_94d14616b488d8d566901bf7a89b5cfd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[72], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_c3e5c99f9354b9c0384bfc3ad7e84868(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94d14616b488d8d566901bf7a89b5cfd
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[72], dtype='int64'),
        ]


class PrimitiveOp_8cdb30031f64a2a6b8e5b5d3801d1913(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[36], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_483336731532f406e97942f8813661ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cdb30031f64a2a6b8e5b5d3801d1913
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[36], dtype='int64'),
        ]


class PrimitiveOp_a091d29ce208b18010f0de10076cd762(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[18], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_96576711f8439389a93c9322892a7a0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a091d29ce208b18010f0de10076cd762
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], dtype='int64').reshape([18]),
        ]


class PrimitiveOp_eb46c588d5c548e47e4c1ebcb155f687(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6804, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_438dabe8c295c54a72a6cc5f73aeef74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb46c588d5c548e47e4c1ebcb155f687
    def get_inputs(self):
        return [
            paddle.uniform([6804, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_438dabe8c295c54a72a6cc5f73aeef74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb46c588d5c548e47e4c1ebcb155f687
    def get_inputs(self):
        return [
            paddle.uniform([6804, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_9bc354cc19c0d684268006ae3711d35e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c5bcd52d2431636790407e23aa65b90
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[64], dtype='int64'),
        ]


class TestPrimitiveOp_284695197f4d3e7a5e8e522951f94155(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c9a909507418b4175eb6ebef5673493
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[32], dtype='int64'),
        ]


class TestPrimitiveOp_66cb74165486cb8db056065a8784ff2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bfbb0d4cd6c81b6f8cfc01ec7a6ea50
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype='int64').reshape([16]),
        ]


class PrimitiveOp_cd06dc072ea25087da606fefa6f6e8fd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5376, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1475682dcc05ca909665e5c3c00eb70b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd06dc072ea25087da606fefa6f6e8fd
    def get_inputs(self):
        return [
            paddle.uniform([5376, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_1475682dcc05ca909665e5c3c00eb70b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd06dc072ea25087da606fefa6f6e8fd
    def get_inputs(self):
        return [
            paddle.uniform([5376, 2], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_5e8e56454554e6011455fec57b1e522b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d396cc72fc1944924f674fdda50abe8
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
        ]


class PrimitiveOp_d5438f81faff01b739697c21922cbc54(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9261], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_961dee125a6f99aabffe2df18a85d08e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5438f81faff01b739697c21922cbc54
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_27593974db80c06957d17e1b912923de(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9261, 4], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_1602ddd96f4452cd4ff46c752df0e6a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27593974db80c06957d17e1b912923de
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 9261, 4], dtype='int32'),
        ]


class TestPrimitiveOp_635c8baedb2659e3019e129a3d752f1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e0d820df191fb74bb4c161cb34c6478
    def get_inputs(self):
        return [
            paddle.to_tensor(18648, dtype='int32').reshape([]),
        ]


class PrimitiveOp_9162f58d5b2be0e62a895ceee831477e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9261, 1], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e82dd9265048d2b5dc5c72de4b193263(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9162f58d5b2be0e62a895ceee831477e
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261, 1], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_da7006d565f58cd99f001a5741c05d23(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 9261, 68], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_25683de4ce569f6cf86cdf09ec68fa2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da7006d565f58cd99f001a5741c05d23
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 9261, 68], dtype='int32'),
        ]


class PrimitiveOp_04d7315e90da591a8cf49a69531e8787(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4662, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_980c1c119cf692e9e9992608ce7f6fc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04d7315e90da591a8cf49a69531e8787
    def get_inputs(self):
        return [
            paddle.uniform([4662, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_47041800532b3ef9114636ddf2266e49(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[4662, 4], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6adb409d7a22621b0e348c85a4524b88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47041800532b3ef9114636ddf2266e49
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[4662, 4], dtype='int64'),
        ]


class PrimitiveOp_34edc944926fad1beb203880b60b3548(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_968983cc2d72c71ab8cf34af13e715dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34edc944926fad1beb203880b60b3548
    def get_inputs(self):
        return [
            paddle.to_tensor(4, dtype='int64').reshape([]),
        ]


class TestPrimitiveOp_7fe9d49f358a8f6c92d330f60d8b5c0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34edc944926fad1beb203880b60b3548
    def get_inputs(self):
        return [
            paddle.to_tensor(7, dtype='int64').reshape([]),
        ]


class PrimitiveOp_d2370c392870d67783e60fe9c0d2ef1a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 7581], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e34bfc7284f58e256189cd5a168872dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2370c392870d67783e60fe9c0d2ef1a
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_e7cdcfe8eeb455f70178643375d5c4c9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 7581, 4], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d6ccc76c8b4e0d68b329339bfb9d1aef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7cdcfe8eeb455f70178643375d5c4c9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 7581, 4], dtype='int32'),
        ]


class TestPrimitiveOp_30ed455e7bf6e79baee29d9f5077107a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e0d820df191fb74bb4c161cb34c6478
    def get_inputs(self):
        return [
            paddle.to_tensor(15428, dtype='int32').reshape([]),
        ]


class PrimitiveOp_b5be88edbe24d7a56618e848d479ffbe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 7581, 1], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_f1279310c29f0e1d0fcd5b014e63eb18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b5be88edbe24d7a56618e848d479ffbe
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581, 1], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_42e11b0251239d654f34a8401a2665a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.bool)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 7581, 68], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_6b02053953fb946a6047aebc19213579(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42e11b0251239d654f34a8401a2665a6
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[1, 7581, 68], dtype='int32'),
        ]


class PrimitiveOp_a5af3806bddcef69ba8edf303d80d22d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3857, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_89b13ef9334455a09984e7faa900431c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5af3806bddcef69ba8edf303d80d22d
    def get_inputs(self):
        return [
            paddle.uniform([3857, 4], dtype='float32', min=-0.5, max=0.5),
        ]


class PrimitiveOp_b5675411410ceb93181bc6d43d9aa945(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3857, 4], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_e78438bba42a5e1d1eb7ea00a4651d5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b5675411410ceb93181bc6d43d9aa945
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[3857, 4], dtype='int64'),
        ]


class PrimitiveOp_57b68ee16ee8d368351b8c7c2a94f8b0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0):
        return paddle._C_ops.cast(input_0, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_21b957458abecf20e21af454104aafd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57b68ee16ee8d368351b8c7c2a94f8b0
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=-0.5, max=0.5),
        ]


class TestPrimitiveOp_483336731532f406e97942f8813661ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cdb30031f64a2a6b8e5b5d3801d1913
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[36], dtype='int64'),
        ]


class TestPrimitiveOp_483336731532f406e97942f8813661ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cdb30031f64a2a6b8e5b5d3801d1913
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=2, shape=[36], dtype='int64'),
        ]




if __name__ == '__main__':
    unittest.main()