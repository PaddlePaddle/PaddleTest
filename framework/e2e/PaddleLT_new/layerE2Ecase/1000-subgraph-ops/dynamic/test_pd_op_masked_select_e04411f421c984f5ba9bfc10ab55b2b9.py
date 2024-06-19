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



class PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.masked_select(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_d2161ae1aacf6fc43547fedd0a5af2aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_d2161ae1aacf6fc43547fedd0a5af2aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024, 4], dtype='int32'), 'bool'),
        ]


class PrimitiveOp_e743a5f7124b44afbb82f41b7faa1b87(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, input_0, input_1):
        return paddle._C_ops.masked_select(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


class TestPrimitiveOp_bfa20083f226ae5281c15ac909d8d5d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e743a5f7124b44afbb82f41b7faa1b87
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_cedb83f14e997729c3372b615d9bed3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024, 68], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_d2161ae1aacf6fc43547fedd0a5af2aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_7d6dc4dfb1cd0f84f22a7bd02359f67c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_7d6dc4dfb1cd0f84f22a7bd02359f67c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_7faa7b71ba59c617c617adbcdd99d24d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e743a5f7124b44afbb82f41b7faa1b87
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_0b87c45b96c2c85007fb48eb8c92a1b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725, 68], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_7d6dc4dfb1cd0f84f22a7bd02359f67c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_7d4489a8a4badff7be16f6dc408580b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 500, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 500, 128], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_7d4489a8a4badff7be16f6dc408580b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 500, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 500, 128], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_da18f53cf38dbbb5d37c7d4eb7f3a4dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_da18f53cf38dbbb5d37c7d4eb7f3a4dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_d6849f3d61c3c61ebbbc9f7148c8fbca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e743a5f7124b44afbb82f41b7faa1b87
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_33a8a4f408f058284e23d8561c7a4f28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 68], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_da18f53cf38dbbb5d37c7d4eb7f3a4dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_f1d8429df8ca044f807454f861f74c10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_f1d8429df8ca044f807454f861f74c10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_3ef6c5b83c7ee8553de1211ccc73eef5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e743a5f7124b44afbb82f41b7faa1b87
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_f59dcac9203f8c2722443a34f50b88f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 76], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 76], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_f1d8429df8ca044f807454f861f74c10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_6776f58ba93cfdb7cd6297c2e353e5fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 2434, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2434, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_6776f58ba93cfdb7cd6297c2e353e5fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 2434, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2434, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_7a787413715f35c5317a5ecdb592f80b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_7a787413715f35c5317a5ecdb592f80b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_385938c3e61197576b49ba455d7e93f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e743a5f7124b44afbb82f41b7faa1b87
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_05f1201ed66af216bac194cee9c52d10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069, 68], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_7a787413715f35c5317a5ecdb592f80b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_da18f53cf38dbbb5d37c7d4eb7f3a4dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_da18f53cf38dbbb5d37c7d4eb7f3a4dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_d6849f3d61c3c61ebbbc9f7148c8fbca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e743a5f7124b44afbb82f41b7faa1b87
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_33a8a4f408f058284e23d8561c7a4f28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 68], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_da18f53cf38dbbb5d37c7d4eb7f3a4dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_8f10ce9399c1cd319665e62346e92921(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_8f10ce9399c1cd319665e62346e92921(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_bf538a01a8a2af55b93b6ad9ab60ad9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e743a5f7124b44afbb82f41b7faa1b87
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_d7e6c4a1e5abe9ba039853d69ea1465f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109, 68], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_8f10ce9399c1cd319665e62346e92921(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_4d22d6fc7d16935ec91e6267b38e7433(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_4d22d6fc7d16935ec91e6267b38e7433(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_9e05d4e3e8d45a2f6df72fcb77f90882(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e743a5f7124b44afbb82f41b7faa1b87
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_efddc9747ba47ec4a31b75e224187344(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100, 68], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_4d22d6fc7d16935ec91e6267b38e7433(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_f1d8429df8ca044f807454f861f74c10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_f1d8429df8ca044f807454f861f74c10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_3ef6c5b83c7ee8553de1211ccc73eef5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e743a5f7124b44afbb82f41b7faa1b87
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_8b61adc0dc3674fe3024f84fbae9445a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 68], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_f1d8429df8ca044f807454f861f74c10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_7d4489a8a4badff7be16f6dc408580b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 500, 128], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 500, 128], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_4cb36ad712c384bb052398e21a98d0f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 8732, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8732, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_4cb36ad712c384bb052398e21a98d0f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 8732, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8732, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_862350a6342b4dd4684f66211d48d560(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_862350a6342b4dd4684f66211d48d560(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_7d6295bcc18a0b534bb5074133f532a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e743a5f7124b44afbb82f41b7faa1b87
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_685b2fb1a3318fca10bcc5db1352c4bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400, 68], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_862350a6342b4dd4684f66211d48d560(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_86fc4c2cd15f042bb10b9294a1bbc1cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_86fc4c2cd15f042bb10b9294a1bbc1cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_1b4f76e4cdf8ad2fc378c0a0b6e550b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e743a5f7124b44afbb82f41b7faa1b87
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_dd4f00a34c53e1eb4ad70acb4aa2651c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261, 68], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_86fc4c2cd15f042bb10b9294a1bbc1cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_37c20fa5d8a0810a44222db1719c2cd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_37c20fa5d8a0810a44222db1719c2cd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581, 4], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_e30a40710583481167eede913aa95954(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e743a5f7124b44afbb82f41b7faa1b87
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_d5a3c3082aaba821b45e4c849c550e5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 68], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581, 68], dtype='int32'), 'bool'),
        ]


class TestPrimitiveOp_37c20fa5d8a0810a44222db1719c2cd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b405d91397e15950cc5ca88d10cdcc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 4], dtype='float32', min=-0.5, max=0.5),
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581, 4], dtype='int32'), 'bool'),
        ]




if __name__ == '__main__':
    unittest.main()