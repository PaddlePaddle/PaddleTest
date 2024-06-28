import os
os.environ['FLAGS_cinn_new_group_scheduler'] = '1'
os.environ['FLAGS_group_schedule_tiling_first'] = '1'
os.environ['FLAGS_enable_pir_api'] = '1'
os.environ['FLAGS_cinn_bucket_compile'] = '1'
import sys
import unittest
import numpy as np
from dataclasses import dataclass
import typing as t

@dataclass
class Stage:
    name: str
    env_vars: t.Dict[str, str]

cinn_stages = [
    Stage(
        name="dynamic_to_static",
        env_vars=dict(
            PADDLE_DEBUG_ENABLE_CINN=False,
            FLAGS_prim_all=False,
            FLAGS_prim_enable_dynamic=False,
        ),
    ),
    Stage(
        name="prim",
        env_vars=dict(
            PADDLE_DEBUG_ENABLE_CINN=False,
            FLAGS_prim_all=True,
            FLAGS_prim_enable_dynamic=True,
        ),
    ),
    Stage(
        name="infer_symbolic",
        env_vars=dict(
            PADDLE_DEBUG_ENABLE_CINN=True,
            FLAGS_prim_all=True,
            FLAGS_prim_enable_dynamic=True,
            FLAGS_use_cinn=False,
            FLAGS_check_infer_symbolic=True,
        ),
    ),
	Stage(
        name="frontend",
        env_vars=dict(
            PADDLE_DEBUG_ENABLE_CINN=True,
            FLAGS_prim_all=True,
            FLAGS_prim_enable_dynamic=True,
            FLAGS_use_cinn=True,
            FLAGS_check_infer_symbolic=False,
            FLAGS_enable_fusion_fallback=True,
        ), 
    ),
    Stage(
        name="backend",
        env_vars=dict(
            PADDLE_DEBUG_ENABLE_CINN=True,
            FLAGS_prim_all=True,
            FLAGS_prim_enable_dynamic=True,
            FLAGS_use_cinn=True,
            FLAGS_check_infer_symbolic=False,
            FLAGS_enable_fusion_fallback=False,
        ), 
    ),
]

def GetCinnStageByName(name):
    for stage in cinn_stages:
        if stage.name == name:
            return stage
    return None

def GetCurrentCinnStage():
    name = os.getenv('PADDLE_DEBUG_CINN_STAGE_NAME')
    if name is None:
        return None
    stage_names = [stage.name for stage in cinn_stages]
    assert name in stage_names, (
        f"PADDLE_DEBUG_CINN_STAGE_NAME should be in {stage_names}"
    )
    return GetCinnStageByName(name)

def GetPrevCinnStage(stage):
    for i in range(1, len(cinn_stages)):
        if stage is cinn_stages[i]:
            return cinn_stages[i - 1]
    return None

def IsCinnStageEnableDiff():
    value = os.getenv('PADDLE_DEBUG_CINN_STAGE_ENABLE_DIFF')
    enabled = value in {
        '1',
        'true',
        'True',
    }
    if enabled:
        assert GetCurrentCinnStage() is not None
    return enabled

last_cinn_stage_exit_code = None
def LastCINNStageFailed():
    global last_cinn_stage_exit_code
    if last_cinn_stage_exit_code is not None:
        return last_cinn_stage_exit_code != 0
    last_stage = GetPrevCinnStage(GetCurrentCinnStage())
    if last_stage is None:
        return False
    env_vars = dict(
        PADDLE_DEBUG_CINN_STAGE_NAME=last_stage.name,
        PADDLE_DEBUG_CINN_STAGE_ENABLE_DIFF='0',
    )
    env_vars_str = " ".join(
        f"{env_var}={value}"
        for env_var, value in env_vars.items()
    )
    last_cinn_stage_exit_code = os.system(
        f"{env_vars_str} {sys.executable} {__file__} > /dev/null 2>&1"
    )
    return last_cinn_stage_exit_code != 0

def SetDefaultEnv(**env_var2value):
    for env_var, value in env_var2value.items():
        if os.getenv(env_var) is None:
            os.environ[env_var] = str(value)

SetDefaultEnv(
    PADDLE_DEBUG_ENABLE_CINN=True,
    FLAGS_enable_pir_api=True,
    FLAGS_prim_all=True,
    FLAGS_prim_enable_dynamic=True,
    FLAGS_use_cinn=False,
    FLAGS_check_infer_symbolic=False,
    FLAGS_enable_fusion_fallback=False,
)

import paddle

def SetEnvVar(env_var2value):
    for env_var, value in env_var2value.items():
        os.environ[env_var] = str(value)
    paddle.set_flags({
        env_var:value
        for env_var, value in env_var2value.items()
        if env_var.startswith('FLAGS_')
    })

if GetCurrentCinnStage() is not None:
    SetEnvVar(GetCurrentCinnStage().env_vars)

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



if not (IsCinnStageEnableDiff() and LastCINNStageFailed()):
    class PrimitiveOp_b9a5ec247f9ccec1ab40e4758f900bcc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.masked_select(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 500, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 500, 128], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a562f112a07e7de75a6e9ce7e725bff3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9a5ec247f9ccec1ab40e4758f900bcc
        def get_inputs(self):
            return [
                paddle.uniform([1, 500, 128], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 500, 128], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_696517326976d8129f12adb66d0296d9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.masked_select(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 1], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_06722bf9f792fbfddac64438c748f93e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696517326976d8129f12adb66d0296d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 8732, 1], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8732, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_a562f112a07e7de75a6e9ce7e725bff3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9a5ec247f9ccec1ab40e4758f900bcc
        def get_inputs(self):
            return [
                paddle.uniform([1, 500, 128], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 500, 128], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_2ea8a6b2ec10a01c179cc27a027d7a76(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.masked_select(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 4], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0a6fc7e91f5c0b867e8441db7eba0c44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ea8a6b2ec10a01c179cc27a027d7a76
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_0a6fc7e91f5c0b867e8441db7eba0c44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ea8a6b2ec10a01c179cc27a027d7a76
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_950fd23ecaadeb5a96398b242cb11171(InstanceTrait, paddle.nn.Layer):
        
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


    class TestPrimitiveOp_cce84b167fc18894d6c2fc7d06210c08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_950fd23ecaadeb5a96398b242cb11171
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_173e517e709307f80ca548a1d76fe0a6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.masked_select(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 68], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2fe9538d7be3f43efafc5acb5c1a3928(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_173e517e709307f80ca548a1d76fe0a6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_0a6fc7e91f5c0b867e8441db7eba0c44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ea8a6b2ec10a01c179cc27a027d7a76
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_6e99ff23e6ebfe7945a82efa8dd744ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ea8a6b2ec10a01c179cc27a027d7a76
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_6e99ff23e6ebfe7945a82efa8dd744ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ea8a6b2ec10a01c179cc27a027d7a76
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_bea466ba289ded935ac5e083a5a0194b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_950fd23ecaadeb5a96398b242cb11171
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_46fd1edc3b0016f43d924a8f4f5a4f88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_173e517e709307f80ca548a1d76fe0a6
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_6e99ff23e6ebfe7945a82efa8dd744ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ea8a6b2ec10a01c179cc27a027d7a76
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_0a6fc7e91f5c0b867e8441db7eba0c44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ea8a6b2ec10a01c179cc27a027d7a76
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_0a6fc7e91f5c0b867e8441db7eba0c44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ea8a6b2ec10a01c179cc27a027d7a76
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_cce84b167fc18894d6c2fc7d06210c08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_950fd23ecaadeb5a96398b242cb11171
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_4a1600c9f66bb975f6fcfff2b4e08704(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.masked_select(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 76], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ea967eeac171b1add24a0208dfb1cdad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4a1600c9f66bb975f6fcfff2b4e08704
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 76], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 76], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_0a6fc7e91f5c0b867e8441db7eba0c44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ea8a6b2ec10a01c179cc27a027d7a76
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_aeb64e296e2c537a06a070315770db68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ea8a6b2ec10a01c179cc27a027d7a76
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_aeb64e296e2c537a06a070315770db68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ea8a6b2ec10a01c179cc27a027d7a76
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_1c69ef67ca29faa78371aa7591f69ab8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_950fd23ecaadeb5a96398b242cb11171
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_fc8503285ccc85504258cf33e1da6a60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_173e517e709307f80ca548a1d76fe0a6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_aeb64e296e2c537a06a070315770db68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ea8a6b2ec10a01c179cc27a027d7a76
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_018b0d0d483aa3dd32c071975d9d7dc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ea8a6b2ec10a01c179cc27a027d7a76
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_018b0d0d483aa3dd32c071975d9d7dc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ea8a6b2ec10a01c179cc27a027d7a76
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_509d28f371c87ce5538706d38cec525a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_950fd23ecaadeb5a96398b242cb11171
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_545c796ff4cb5c67a5891b433d5904d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_173e517e709307f80ca548a1d76fe0a6
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_018b0d0d483aa3dd32c071975d9d7dc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ea8a6b2ec10a01c179cc27a027d7a76
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_68c9b7283662024e2871447b3f9b31d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ea8a6b2ec10a01c179cc27a027d7a76
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_68c9b7283662024e2871447b3f9b31d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ea8a6b2ec10a01c179cc27a027d7a76
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_afd834a48ebc21525b092c8f85d55026(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_950fd23ecaadeb5a96398b242cb11171
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_3bc97484076bab4c17d7ee4a356d51c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_173e517e709307f80ca548a1d76fe0a6
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_68c9b7283662024e2871447b3f9b31d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ea8a6b2ec10a01c179cc27a027d7a76
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261, 4], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6(InstanceTrait, paddle.nn.Layer):
        
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


    class TestPrimitiveOp_746f94f7499b0c7351ac5f4cd63909e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 2434, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2434, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_746f94f7499b0c7351ac5f4cd63909e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 2434, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2434, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_c5fe5c1ec16f73f0414f2245a679cd5f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696517326976d8129f12adb66d0296d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 2434, 1], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2434, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_10fca638f29e108a52f90b9691154f2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ea8a6b2ec10a01c179cc27a027d7a76
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_10fca638f29e108a52f90b9691154f2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ea8a6b2ec10a01c179cc27a027d7a76
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_f1cba9908978eaaf57f01f2d081dd2fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_950fd23ecaadeb5a96398b242cb11171
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_7ed8b692d55729b1eca05ba3fbb254ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_173e517e709307f80ca548a1d76fe0a6
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_10fca638f29e108a52f90b9691154f2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ea8a6b2ec10a01c179cc27a027d7a76
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_4b4d12fc64a2d361fa6499be5b30866d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ea8a6b2ec10a01c179cc27a027d7a76
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_4b4d12fc64a2d361fa6499be5b30866d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ea8a6b2ec10a01c179cc27a027d7a76
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_32d53feb369b6b10d1146be0da8e547f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_950fd23ecaadeb5a96398b242cb11171
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_f8398e85b0bbefb75e2f8401ba7f83d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_173e517e709307f80ca548a1d76fe0a6
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_4b4d12fc64a2d361fa6499be5b30866d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ea8a6b2ec10a01c179cc27a027d7a76
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_6c9315a609f1dac9cde620a042f36b10(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ea8a6b2ec10a01c179cc27a027d7a76
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_6c9315a609f1dac9cde620a042f36b10(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ea8a6b2ec10a01c179cc27a027d7a76
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_0c0c130782f116c692be3f120976dee3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_950fd23ecaadeb5a96398b242cb11171
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_0f60a954bcd1bb7bb055cc224a055331(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_173e517e709307f80ca548a1d76fe0a6
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_6c9315a609f1dac9cde620a042f36b10(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ea8a6b2ec10a01c179cc27a027d7a76
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_26a78cd2ab387e722b87de1d502487c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ea8a6b2ec10a01c179cc27a027d7a76
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_26a78cd2ab387e722b87de1d502487c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ea8a6b2ec10a01c179cc27a027d7a76
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_1df5875bb634aafd8a44ccbd44cf650c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_950fd23ecaadeb5a96398b242cb11171
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_a85962b328093573be5a36849f0f3ce5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_173e517e709307f80ca548a1d76fe0a6
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_26a78cd2ab387e722b87de1d502487c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ea8a6b2ec10a01c179cc27a027d7a76
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_117ee13a48b20b469d3b7e89b0cd85f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 8732, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8732, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_117ee13a48b20b469d3b7e89b0cd85f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 8732, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8732, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_018b0d0d483aa3dd32c071975d9d7dc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ea8a6b2ec10a01c179cc27a027d7a76
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_018b0d0d483aa3dd32c071975d9d7dc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ea8a6b2ec10a01c179cc27a027d7a76
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_509d28f371c87ce5538706d38cec525a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_950fd23ecaadeb5a96398b242cb11171
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_545c796ff4cb5c67a5891b433d5904d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_173e517e709307f80ca548a1d76fe0a6
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_018b0d0d483aa3dd32c071975d9d7dc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ea8a6b2ec10a01c179cc27a027d7a76
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_a562f112a07e7de75a6e9ce7e725bff3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9a5ec247f9ccec1ab40e4758f900bcc
        def get_inputs(self):
            return [
                paddle.uniform([1, 500, 128], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 500, 128], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_b25463fca7c33b4950ee38e573c5b562(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ea8a6b2ec10a01c179cc27a027d7a76
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_b25463fca7c33b4950ee38e573c5b562(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ea8a6b2ec10a01c179cc27a027d7a76
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_039adecd7de128608dc4cc723e73eff4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_950fd23ecaadeb5a96398b242cb11171
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_9ea8e401cfd96a4deb3f88762b626146(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_173e517e709307f80ca548a1d76fe0a6
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_b25463fca7c33b4950ee38e573c5b562(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ea8a6b2ec10a01c179cc27a027d7a76
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_a562f112a07e7de75a6e9ce7e725bff3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9a5ec247f9ccec1ab40e4758f900bcc
        def get_inputs(self):
            return [
                paddle.uniform([1, 500, 128], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 500, 128], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_a36469cb057a0d60649a7c85af5083ff(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.masked_select(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8732, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 8732, 1], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9a3bad6965814b794402926bc49d0982(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a36469cb057a0d60649a7c85af5083ff
        def get_inputs(self):
            return [
                paddle.uniform([1, 8732, 1], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8732, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_a562f112a07e7de75a6e9ce7e725bff3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9a5ec247f9ccec1ab40e4758f900bcc
        def get_inputs(self):
            return [
                paddle.uniform([1, 500, 128], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 500, 128], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_577897ed58e167d74a3c77bffd7eea84(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.masked_select(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3549, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3549, 4], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_83be0e34200986cfebbad67601290ee5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_577897ed58e167d74a3c77bffd7eea84
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_83be0e34200986cfebbad67601290ee5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_577897ed58e167d74a3c77bffd7eea84
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_55c9ff9e85c0b360497ebee8f30d98c7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.masked_select(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3549], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3549], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3344a3a7e239ea175c61cb2ebfe7a915(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55c9ff9e85c0b360497ebee8f30d98c7
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_d85575546664acff1f1c5ee66be0dcbd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.masked_select(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3549, 68], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3549, 68], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_670cfcfcbb5d14e536327c1e6d45506f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d85575546664acff1f1c5ee66be0dcbd
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_83be0e34200986cfebbad67601290ee5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_577897ed58e167d74a3c77bffd7eea84
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_78c92eba0c99009b63df18753541d2cc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.masked_select(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 11109, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 11109, 4], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_94e3988019a6dd3fddc710e8f8d23f3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78c92eba0c99009b63df18753541d2cc
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_94e3988019a6dd3fddc710e8f8d23f3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78c92eba0c99009b63df18753541d2cc
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109, 4], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_0fb35f63091bc763d66cd07690609243(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.masked_select(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 11109], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 11109], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_873970f1a5b7abeb632045020d002a18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0fb35f63091bc763d66cd07690609243
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_d2014ee21b18fde25a09a532e2b66362(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.masked_select(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 11109, 68], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 11109, 68], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_29099791819eae2d0aad762e8f613070(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d2014ee21b18fde25a09a532e2b66362
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_94e3988019a6dd3fddc710e8f8d23f3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78c92eba0c99009b63df18753541d2cc
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_83be0e34200986cfebbad67601290ee5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_577897ed58e167d74a3c77bffd7eea84
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_83be0e34200986cfebbad67601290ee5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_577897ed58e167d74a3c77bffd7eea84
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_3344a3a7e239ea175c61cb2ebfe7a915(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55c9ff9e85c0b360497ebee8f30d98c7
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_3a648bdc1e054b4aa7e3490b28737866(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.masked_select(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3549, 76], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3549, 76], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_984408dc35fc9d806d5f8b6f66396f9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3a648bdc1e054b4aa7e3490b28737866
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 76], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 76], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_83be0e34200986cfebbad67601290ee5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_577897ed58e167d74a3c77bffd7eea84
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_e14bfdb74d07503fb94bb293cc8c91b4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.masked_select(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3024, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3024, 4], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8b16f8752345a74c429eb4a03000bda5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e14bfdb74d07503fb94bb293cc8c91b4
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_8b16f8752345a74c429eb4a03000bda5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e14bfdb74d07503fb94bb293cc8c91b4
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024, 4], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_1414d7a26bb3ccc7c7e29acf2ee2932c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.masked_select(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3024], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3024], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_68b7896c8aacd6f9389531c0f06ed5cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1414d7a26bb3ccc7c7e29acf2ee2932c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_31ef10e0a2847a4a6ccc93531780df3b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.masked_select(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3024, 68], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3024, 68], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a525075cebca743406bdadb56e5a2d9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_31ef10e0a2847a4a6ccc93531780df3b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_8b16f8752345a74c429eb4a03000bda5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e14bfdb74d07503fb94bb293cc8c91b4
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024, 4], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_43dca1fbb29b274afb4a7640a83cd961(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.masked_select(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4116, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 4116, 4], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ad4d5d2f09a073334ce24dbec946d846(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43dca1fbb29b274afb4a7640a83cd961
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_ad4d5d2f09a073334ce24dbec946d846(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43dca1fbb29b274afb4a7640a83cd961
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_1ce590e34fbc91731cb5a25053701d09(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.masked_select(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4116], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 4116], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_776b60fab656f962071ed1f5f24a23d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ce590e34fbc91731cb5a25053701d09
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_58d5f40a9725116106f4c5f9e70e6716(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.masked_select(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4116, 68], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 4116, 68], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3693b653783bdb684a3073f5e359be7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_58d5f40a9725116106f4c5f9e70e6716
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_ad4d5d2f09a073334ce24dbec946d846(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43dca1fbb29b274afb4a7640a83cd961
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_d4a4bdf8331832668221d7eab2672eed(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.masked_select(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 9261, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 9261, 4], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b5f37c7a7c4eddbec55b4f973e60a903(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4a4bdf8331832668221d7eab2672eed
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_b5f37c7a7c4eddbec55b4f973e60a903(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4a4bdf8331832668221d7eab2672eed
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261, 4], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_8c52aaf964991a0a7d2dd0e5ff3eaccb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.masked_select(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 9261], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 9261], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8f70f70f49f57f0bce8fdbaec9f60844(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c52aaf964991a0a7d2dd0e5ff3eaccb
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_f3ff8efda31409adc6468fe5ef3ca077(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.masked_select(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 9261, 68], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 9261, 68], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8ee4504850b5f4b35198731dfdfb61e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f3ff8efda31409adc6468fe5ef3ca077
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_b5f37c7a7c4eddbec55b4f973e60a903(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4a4bdf8331832668221d7eab2672eed
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261, 4], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_98772737cc6a5ac568c1f842d56ba785(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.masked_select(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2434, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 2434, 4], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_de7e377f80c23e95eb370dee28746c3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98772737cc6a5ac568c1f842d56ba785
        def get_inputs(self):
            return [
                paddle.uniform([1, 2434, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2434, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_de7e377f80c23e95eb370dee28746c3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98772737cc6a5ac568c1f842d56ba785
        def get_inputs(self):
            return [
                paddle.uniform([1, 2434, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2434, 4], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_10bf964dccf2745ebacc532e83d08d02(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.masked_select(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2434, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 2434, 1], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_703ddacde0b0938be4938d9c17280be6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_10bf964dccf2745ebacc532e83d08d02
        def get_inputs(self):
            return [
                paddle.uniform([1, 2434, 1], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2434, 1], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_90cc852bc263bc1743871efb77595137(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.masked_select(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2100, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 2100, 4], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_21b9a1d7f5ce6cbca1f29603506d0959(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90cc852bc263bc1743871efb77595137
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_21b9a1d7f5ce6cbca1f29603506d0959(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90cc852bc263bc1743871efb77595137
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100, 4], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_6f5d6026afa4999a7dde4df371f51d5c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.masked_select(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2100], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 2100], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8522f0cd28a9fd8849e9af374ee4ff40(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f5d6026afa4999a7dde4df371f51d5c
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_f9a288883ab386dcbcdff18298c4aeb1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.masked_select(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2100, 68], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 2100, 68], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0fb2c4852ec6be56567c36ad6a1d37f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9a288883ab386dcbcdff18298c4aeb1
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_21b9a1d7f5ce6cbca1f29603506d0959(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_90cc852bc263bc1743871efb77595137
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100, 4], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_6c29daf15c1c520fddc7c51e7886342f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.masked_select(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4725, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 4725, 4], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6a6af2052d38259357b622e54f8678fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6c29daf15c1c520fddc7c51e7886342f
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_6a6af2052d38259357b622e54f8678fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6c29daf15c1c520fddc7c51e7886342f
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725, 4], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_590871b9008bcce275fd45442a2eb2b1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.masked_select(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4725], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 4725], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_138004d2078abf895e6b3f7c37f2f48a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_590871b9008bcce275fd45442a2eb2b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_23d9a58cf2e0792db95245d81c05bc98(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.masked_select(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4725, 68], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 4725, 68], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6a8295bf0c94a87d57da7acd5f3dc6ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_23d9a58cf2e0792db95245d81c05bc98
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_6a6af2052d38259357b622e54f8678fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6c29daf15c1c520fddc7c51e7886342f
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725, 4], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_c6a4aa0df59cb9ed65b6abdde1971ca3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.masked_select(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 6069, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 6069, 4], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_53e4bb0e0fa28393ac8069999be2f59e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6a4aa0df59cb9ed65b6abdde1971ca3
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_53e4bb0e0fa28393ac8069999be2f59e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6a4aa0df59cb9ed65b6abdde1971ca3
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069, 4], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_62d290edcb62e3258d111b6369d0403a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.masked_select(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 6069], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 6069], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8cb3ad9770a503b2d1b1cd0e1ea5c416(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_62d290edcb62e3258d111b6369d0403a
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_af12865bd5b8b5bf00e7500e1291a06b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.masked_select(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 6069, 68], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 6069, 68], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_eb22d913466550381786c03761e708a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_af12865bd5b8b5bf00e7500e1291a06b
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_53e4bb0e0fa28393ac8069999be2f59e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6a4aa0df59cb9ed65b6abdde1971ca3
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069, 4], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_382f364addb41d5e517eeb564b10e5f7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.masked_select(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 7581, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 7581, 4], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d18da9ddd7712c17edd89aee94404cb4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_382f364addb41d5e517eeb564b10e5f7
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_d18da9ddd7712c17edd89aee94404cb4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_382f364addb41d5e517eeb564b10e5f7
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581, 4], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_cfb266a0af45b74f90372c9d9f1f31b9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.masked_select(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 7581], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 7581], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_30993e26e1ae6c7d41c6cd15cc48736c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cfb266a0af45b74f90372c9d9f1f31b9
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_6f11170aeadbecd5897d475d22915954(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.masked_select(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 7581, 68], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 7581, 68], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e0ec8ff71036c60eacffe28f3d7346ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f11170aeadbecd5897d475d22915954
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_d18da9ddd7712c17edd89aee94404cb4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_382f364addb41d5e517eeb564b10e5f7
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581, 4], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_838d8a64efc5aae5081ef04592d334ee(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.masked_select(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8732, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 8732, 4], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e997748d8145e5875fb2947d43a01872(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_838d8a64efc5aae5081ef04592d334ee
        def get_inputs(self):
            return [
                paddle.uniform([1, 8732, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8732, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_e997748d8145e5875fb2947d43a01872(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_838d8a64efc5aae5081ef04592d334ee
        def get_inputs(self):
            return [
                paddle.uniform([1, 8732, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8732, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_ad4d5d2f09a073334ce24dbec946d846(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43dca1fbb29b274afb4a7640a83cd961
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_ad4d5d2f09a073334ce24dbec946d846(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43dca1fbb29b274afb4a7640a83cd961
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_776b60fab656f962071ed1f5f24a23d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ce590e34fbc91731cb5a25053701d09
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_3693b653783bdb684a3073f5e359be7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_58d5f40a9725116106f4c5f9e70e6716
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_ad4d5d2f09a073334ce24dbec946d846(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43dca1fbb29b274afb4a7640a83cd961
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_a562f112a07e7de75a6e9ce7e725bff3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9a5ec247f9ccec1ab40e4758f900bcc
        def get_inputs(self):
            return [
                paddle.uniform([1, 500, 128], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 500, 128], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_e9d37fab9f0d7538369325910208c56d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.masked_select(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8400, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 8400, 4], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5b68c1ce88d039d61d8f68861a4560e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e9d37fab9f0d7538369325910208c56d
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_5b68c1ce88d039d61d8f68861a4560e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e9d37fab9f0d7538369325910208c56d
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400, 4], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_08a2cdfc8b79e5772dd31cd8de5e76ef(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.masked_select(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8400], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 8400], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6655d2071a9cc27b737595b412f0514d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08a2cdfc8b79e5772dd31cd8de5e76ef
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_98e86e081709fa198c077f7c58ffa293(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.masked_select(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8400, 68], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 8400, 68], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_08f4b2b62f9c791bdde8a576bbcaaaa8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98e86e081709fa198c077f7c58ffa293
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_5b68c1ce88d039d61d8f68861a4560e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e9d37fab9f0d7538369325910208c56d
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_f4dea16827370003b67c9bc4acf3ff81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 500, 128], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 500, 128], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_0b8a7f5320f4370e84ad2722819fe64a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 8732, 1], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8732, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_f4dea16827370003b67c9bc4acf3ff81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 500, 128], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 500, 128], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_3862ec43d4f9346c627fdee9d53e4fd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_3862ec43d4f9346c627fdee9d53e4fd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_cce84b167fc18894d6c2fc7d06210c08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_950fd23ecaadeb5a96398b242cb11171
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_c69a5375912b5d7b8b12aec1b9ba078c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_3862ec43d4f9346c627fdee9d53e4fd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_9feeaf5a3904cece836f32050bb077ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_9feeaf5a3904cece836f32050bb077ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_bea466ba289ded935ac5e083a5a0194b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_950fd23ecaadeb5a96398b242cb11171
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_52802e6fe3155469f96f1e405d245bf6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_9feeaf5a3904cece836f32050bb077ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_3862ec43d4f9346c627fdee9d53e4fd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_3862ec43d4f9346c627fdee9d53e4fd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_cce84b167fc18894d6c2fc7d06210c08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_950fd23ecaadeb5a96398b242cb11171
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_96363917d1e32a8773041ccdc0fe4c39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 76], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 76], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_3862ec43d4f9346c627fdee9d53e4fd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_07f4b53d75c7acd41cca54a1274c65ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_07f4b53d75c7acd41cca54a1274c65ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_1c69ef67ca29faa78371aa7591f69ab8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_950fd23ecaadeb5a96398b242cb11171
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_7cd4feb60040602710099a76801f71b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_07f4b53d75c7acd41cca54a1274c65ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_5ab2e8c1868caa3ab05cb0902b2e9655(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_5ab2e8c1868caa3ab05cb0902b2e9655(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_509d28f371c87ce5538706d38cec525a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_950fd23ecaadeb5a96398b242cb11171
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_8c536f840acfafbfaba59ffbf42c9aa6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_5ab2e8c1868caa3ab05cb0902b2e9655(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_9d988304a3a62f1bbd24225faeeb9785(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_9d988304a3a62f1bbd24225faeeb9785(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_afd834a48ebc21525b092c8f85d55026(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_950fd23ecaadeb5a96398b242cb11171
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_c74700bd7bb2aebcec6926a76a0523d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_9d988304a3a62f1bbd24225faeeb9785(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_746f94f7499b0c7351ac5f4cd63909e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 2434, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2434, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_746f94f7499b0c7351ac5f4cd63909e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 2434, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2434, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_00d5ffeb1d0cada6cfc085b9483e3fa8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 2434, 1], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2434, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_0ea05636dfe2de8433d74cb9275657c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_0ea05636dfe2de8433d74cb9275657c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_f1cba9908978eaaf57f01f2d081dd2fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_950fd23ecaadeb5a96398b242cb11171
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_1fefcc25cbbdbe5ef83f6b8e333ccbb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_0ea05636dfe2de8433d74cb9275657c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_22668104d02a172d3b9cfdc17fa87162(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_22668104d02a172d3b9cfdc17fa87162(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_32d53feb369b6b10d1146be0da8e547f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_950fd23ecaadeb5a96398b242cb11171
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_da0a6b57a2d8cc72ab1e637a48c4f82d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_22668104d02a172d3b9cfdc17fa87162(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_c6a50ed1dc6b0cf28372d7aebd726722(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_c6a50ed1dc6b0cf28372d7aebd726722(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_0c0c130782f116c692be3f120976dee3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_950fd23ecaadeb5a96398b242cb11171
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_4894d5da82b68663c718868dd378da57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_c6a50ed1dc6b0cf28372d7aebd726722(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_c92f3b123e6dc66f3bf75f85c1142e0f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_c92f3b123e6dc66f3bf75f85c1142e0f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_1df5875bb634aafd8a44ccbd44cf650c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_950fd23ecaadeb5a96398b242cb11171
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_5c8d04d80e77cd353d083cd8d9cee20e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_c92f3b123e6dc66f3bf75f85c1142e0f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_117ee13a48b20b469d3b7e89b0cd85f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 8732, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8732, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_117ee13a48b20b469d3b7e89b0cd85f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 8732, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8732, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_5ab2e8c1868caa3ab05cb0902b2e9655(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_5ab2e8c1868caa3ab05cb0902b2e9655(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_509d28f371c87ce5538706d38cec525a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_950fd23ecaadeb5a96398b242cb11171
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_8c536f840acfafbfaba59ffbf42c9aa6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_5ab2e8c1868caa3ab05cb0902b2e9655(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_f4dea16827370003b67c9bc4acf3ff81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 500, 128], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 500, 128], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_8c07eb8a6ae82960ca8f8cfabb662aad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_8c07eb8a6ae82960ca8f8cfabb662aad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_039adecd7de128608dc4cc723e73eff4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_950fd23ecaadeb5a96398b242cb11171
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_f57c57cc463324a0187f5d3ff7feba4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_8c07eb8a6ae82960ca8f8cfabb662aad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944c0fe3fc30313f8d7afca10a7691f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400, 4], dtype='int32'), 'bool'),
            ]


    

if __name__ == '__main__':
    unittest.main()