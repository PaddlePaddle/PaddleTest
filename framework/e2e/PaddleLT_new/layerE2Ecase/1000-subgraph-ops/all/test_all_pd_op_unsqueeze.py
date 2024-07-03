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
            PADDLE_DEBUG_ENABLE_CINN=False,
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
    PADDLE_DEBUG_CINN_STAGE_NAME="backend",
    PADDLE_DEBUG_CINN_STAGE_ENABLE_DIFF=False,
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
        paddle.seed(2024)
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





last_stage_failed = (IsCinnStageEnableDiff() and LastCINNStageFailed())
class PrimitiveOp_64742105392b863a7fe416d40d6b56c8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0, 1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_298a3c3df4c2bd7166d5812d103bfaa5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64742105392b863a7fe416d40d6b56c8
    def get_inputs(self):
        return [
            paddle.uniform([24, 36], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_298a3c3df4c2bd7166d5812d103bfaa5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64742105392b863a7fe416d40d6b56c8
    def get_inputs(self):
        return [
            paddle.uniform([24, 36], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_edd45e5c1310601f8caf21562497671a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 72], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_49143326b68e1bb27456a508fcfd3a02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edd45e5c1310601f8caf21562497671a
    def get_inputs(self):
        return [
            paddle.uniform([1, 72], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_794d36fe53e387ae8f7df45434e5ab56(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 92], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7a3e1763b4bc46e95ad3aab98d84cdac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_794d36fe53e387ae8f7df45434e5ab56
    def get_inputs(self):
        return [
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_75d134f46ed31accb5261c267588ed9e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 960], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e62e8b7be10497a950f965094108f146(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75d134f46ed31accb5261c267588ed9e
    def get_inputs(self):
        return [
            paddle.uniform([1, 960], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_9d48269ce2eaabfb5601e3f759c1fa17(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 480], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7396d1336cb551e74579b595f617afdd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d48269ce2eaabfb5601e3f759c1fa17
    def get_inputs(self):
        return [
            paddle.uniform([1, 480], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_d67cbf60ee7a96c433e41952291ebe5a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 336], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c1871c05b3385ba3b5c683dd9ca5042d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d67cbf60ee7a96c433e41952291ebe5a
    def get_inputs(self):
        return [
            paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_c8e8b8c14d9af9d159aea840edf98b23(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b9c14da4b5e43ebd43cc141d59a61ebe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8e8b8c14d9af9d159aea840edf98b23
    def get_inputs(self):
        return [
            paddle.uniform([4, 80], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_475cb32d1c927b9d08f8e63628700a91(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1, 2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_59613c9d825c01ba2fb468a520b1b52e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_475cb32d1c927b9d08f8e63628700a91
    def get_inputs(self):
        return [
            paddle.to_tensor([0.33804893493652344, 0.39942264556884766, 0.09662508219480515, 0.1508401781320572], dtype='float32').reshape([4]),
        ]



class PrimitiveOp_5d1704b5b1562e3ba7f123ecf4ab02d9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 2, 9, 112, 112], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fec6235fe33133c68ad99b8b1839b919(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d1704b5b1562e3ba7f123ecf4ab02d9
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 9, 112, 112], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_897cc9a673b22b3e799dea2fcfbb6f08(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4c8149e3286a7df3e8d7e6ffa3a7ca12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_897cc9a673b22b3e799dea2fcfbb6f08
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_77dcac572ff7d24b9e7ddf7971c603e8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_28eb6c514b1336191185c77482b988fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77dcac572ff7d24b9e7ddf7971c603e8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 500], dtype='int64'),
        ]



class PrimitiveOp_0ea9493df868c15c17fe1346efc13bbb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b139dca4b759021550f9c05fd13dae78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ea9493df868c15c17fe1346efc13bbb
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 500], dtype='int32'),
        ]



class PrimitiveOp_f42c594134187d886dbe754f9d7c3b5b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_78b89bad929706d58b4b8d3d96ca1c7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f42c594134187d886dbe754f9d7c3b5b
    def get_inputs(self):
        return [
            paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_6aff6f421b758d1590f0e3f7805ed276(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d21e1fe8c6c3aa891b82e1359054f576(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aff6f421b758d1590f0e3f7805ed276
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8732], dtype='int32'), 'bool'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4526a69f73f785f9055b446aa40a9b78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d67cbf60ee7a96c433e41952291ebe5a
    def get_inputs(self):
        return [
            paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_dd329f5fde19fcac72c8c3ae628c0049(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_51651f4c48c5a43e815e355416c199b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd329f5fde19fcac72c8c3ae628c0049
    def get_inputs(self):
        return [
            paddle.uniform([21, 256], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_28eb6c514b1336191185c77482b988fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77dcac572ff7d24b9e7ddf7971c603e8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 500], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b139dca4b759021550f9c05fd13dae78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ea9493df868c15c17fe1346efc13bbb
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 500], dtype='int32'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4526a69f73f785f9055b446aa40a9b78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d67cbf60ee7a96c433e41952291ebe5a
    def get_inputs(self):
        return [
            paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_0d32ca250ce5245c2f823d77c4185bbb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5993e87c50c296a408bdc04991c44233(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d32ca250ce5245c2f823d77c4185bbb
    def get_inputs(self):
        return [
            paddle.uniform([1, 21824, 2], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_2f0d6bc32e9943d2d3694cb97215d4c4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_02ad572044a2cfb4b003d96bd41a603f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f0d6bc32e9943d2d3694cb97215d4c4
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.29133909940719604, 0.4853375554084778], [0.36761757731437683, 0.4571003019809723], [0.46794551610946655, 0.1688530147075653], [0.2670198976993561, 0.1982056200504303], [0.12026780098676682, 0.48663008213043213], [0.2773101329803467, 0.48897823691368103]]], dtype='float32').reshape([1, 6, 2]),
        ]



class PrimitiveOp_a29295ec635497b7fdbf6b4658c1c547(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 4, 49, 56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4735d5777911f9707e8f54383ad87253(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a29295ec635497b7fdbf6b4658c1c547
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 49, 56, 56], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_b187efd56361e4022b806203fc001de9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1a1be7b66172e1787d2855997c39f2ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b187efd56361e4022b806203fc001de9
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([16]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ce05199e78ede71a9dca2db88cb9d450(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b187efd56361e4022b806203fc001de9
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([16]),
        ]



class PrimitiveOp_b007640545457cf3480b8ca11eafc997(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_44406fca3fafa8f26f191b33b364160d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b007640545457cf3480b8ca11eafc997
    def get_inputs(self):
        return [
            paddle.uniform([145, 240], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_a8fb7d04d3d0a82dc1a4a637cedf9337(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, 19], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_287119df377c0c7844ff4561fa2c1fa8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8fb7d04d3d0a82dc1a4a637cedf9337
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_55befd650fdc9fa9414195dbf32cea1a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6c65f66afc5bb1ca9fbc53e4a1cd3586(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55befd650fdc9fa9414195dbf32cea1a
    def get_inputs(self):
        return [
            paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_de692023e6733e944677e99243258156(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3cb823eb7dfa307fecdec8329746ff1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de692023e6733e944677e99243258156
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.18305830657482147, 0.25759702920913696, 0.46923521161079407, 0.37422502040863037]], dtype='float32').reshape([1, 4]),
        ]



class PrimitiveOp_3fa62432cbd1c90bed002287af07b5a4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e52056eee7d3b9c92c335a86fe2cdd24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3fa62432cbd1c90bed002287af07b5a4
    def get_inputs(self):
        return [
            paddle.uniform([300, 256], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_2315886efe1966cb990875ac0ccefc79(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_63eba50cf16fdf1050dc069ab09caf3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2315886efe1966cb990875ac0ccefc79
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 64, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_63eba50cf16fdf1050dc069ab09caf3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2315886efe1966cb990875ac0ccefc79
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 64, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b0991d71d26094399cd2cc751313b060(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d32ca250ce5245c2f823d77c4185bbb
    def get_inputs(self):
        return [
            paddle.uniform([512, 64, 128], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_4cc8b80e88e29763cacea0fbc9e96315(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e5567679dfe0c86329e936537f5ccbe6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4cc8b80e88e29763cacea0fbc9e96315
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fd132e51864e27315836cdba6aa9771e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64742105392b863a7fe416d40d6b56c8
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fd132e51864e27315836cdba6aa9771e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64742105392b863a7fe416d40d6b56c8
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_f284da1dcfce1be739ce25791dea96c6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, 21], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7ed5587ca9ed53e30208def535013e06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f284da1dcfce1be739ce25791dea96c6
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 21], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0db7012b0f352137c93f769a63e6e5af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8e8b8c14d9af9d159aea840edf98b23
    def get_inputs(self):
        return [
            paddle.uniform([3, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c63f20bf2174b4e82ae2ac12a1b35723(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_475cb32d1c927b9d08f8e63628700a91
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2647590637207031, 0.05532848462462425, 0.46784508228302], dtype='float32').reshape([3]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e3f6644f5446254f13fab799b122e971(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f42c594134187d886dbe754f9d7c3b5b
    def get_inputs(self):
        return [
            paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_ee828dc99c1f501282115dbef8bec917(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 872], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_afe9f2f9e806ecf43943f8f7e30d0bef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee828dc99c1f501282115dbef8bec917
    def get_inputs(self):
        return [
            paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_e796214aa571e77649fd59840cfd70ea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4cb59ce08bafe761551caf2cb3e3c893(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e796214aa571e77649fd59840cfd70ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
        ]



class PrimitiveOp_e641ca14c54a3f9fc8693dce5c5c1b27(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_73676d7b6b3608fc76c99a5ab8883279(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e641ca14c54a3f9fc8693dce5c5c1b27
    def get_inputs(self):
        return [
            paddle.uniform([1777], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_eea885eda6722405ec5b3fe2cd6ca875(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aff6f421b758d1590f0e3f7805ed276
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
        ]



class PrimitiveOp_4d02bec4744d38c009b4a112d792b01a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_458f3aa7d00029759f2546878940d951(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d02bec4744d38c009b4a112d792b01a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1777, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_458f3aa7d00029759f2546878940d951(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d02bec4744d38c009b4a112d792b01a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1777, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d1ba08425efedfd97862a878d318558d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64742105392b863a7fe416d40d6b56c8
    def get_inputs(self):
        return [
            paddle.uniform([20, 30], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d1ba08425efedfd97862a878d318558d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64742105392b863a7fe416d40d6b56c8
    def get_inputs(self):
        return [
            paddle.uniform([20, 30], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_9784e251c6e38d21561db8e4da41c24b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 4, 49, 56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4a537c4c21994a6a942d40a8ad1f9e31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9784e251c6e38d21561db8e4da41c24b
    def get_inputs(self):
        return [
            paddle.uniform([22, 4, 49, 56, 56], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7566cdf98dee30d7ec78c7b1be431163(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d67cbf60ee7a96c433e41952291ebe5a
    def get_inputs(self):
        return [
            paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_65abb7c908c329c7eb1679069b618179(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 768, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_278c7543ed909cea69c42e33ac11c271(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65abb7c908c329c7eb1679069b618179
    def get_inputs(self):
        return [
            paddle.uniform([43, 768, 49], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7a74a8aa0f13d50ebde0c2f296870f36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b007640545457cf3480b8ca11eafc997
    def get_inputs(self):
        return [
            paddle.uniform([10, 240], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3866202279d18be06b70d7021539861f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e796214aa571e77649fd59840cfd70ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 11109], dtype='int32'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bf224eec18902dcb00eeb4ab5a04afd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e641ca14c54a3f9fc8693dce5c5c1b27
    def get_inputs(self):
        return [
            paddle.uniform([5480], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b4133880239758627b547f5798c630b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aff6f421b758d1590f0e3f7805ed276
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109], dtype='int32'), 'bool'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_282c5cdd227632c619230b4dbca0efdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d02bec4744d38c009b4a112d792b01a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[5480, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_282c5cdd227632c619230b4dbca0efdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d02bec4744d38c009b4a112d792b01a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[5480, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_81859d329c659b73fd9a030e4a33d212(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b187efd56361e4022b806203fc001de9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[36], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_81859d329c659b73fd9a030e4a33d212(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b187efd56361e4022b806203fc001de9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[36], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c1871c05b3385ba3b5c683dd9ca5042d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d67cbf60ee7a96c433e41952291ebe5a
    def get_inputs(self):
        return [
            paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_4bb80a75edd5f8907b3cdb9307144b45(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 32, 49, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8ff67d519e5fbeba0c3c00a573ef9f28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb80a75edd5f8907b3cdb9307144b45
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 49, 7, 7], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6dc24f4cb87d059a221da5164e3c82f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd329f5fde19fcac72c8c3ae628c0049
    def get_inputs(self):
        return [
            paddle.uniform([19, 256], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_5dea7894d6cc15a393a5335c4b5dbb7e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_df32c6a6306154ecba9cfecd0ae4ad8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dea7894d6cc15a393a5335c4b5dbb7e
    def get_inputs(self):
        return [
            paddle.uniform([10, 36], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ab9340c49b9aa9a60559e3f931f5fd0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d48269ce2eaabfb5601e3f759c1fa17
    def get_inputs(self):
        return [
            paddle.uniform([10, 480], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4cb59ce08bafe761551caf2cb3e3c893(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e796214aa571e77649fd59840cfd70ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2ddf54be1dddfaa8d15d815adb5f9a23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e641ca14c54a3f9fc8693dce5c5c1b27
    def get_inputs(self):
        return [
            paddle.uniform([1742], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_eea885eda6722405ec5b3fe2cd6ca875(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aff6f421b758d1590f0e3f7805ed276
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c6444fd10b3f2b5d73997034fea044c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d02bec4744d38c009b4a112d792b01a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1742, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c6444fd10b3f2b5d73997034fea044c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d02bec4744d38c009b4a112d792b01a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1742, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_423212f76b26b237aa560b02cb690478(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d67cbf60ee7a96c433e41952291ebe5a
    def get_inputs(self):
        return [
            paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8ddd3f118c705c4e6ca49e452ea4f861(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b007640545457cf3480b8ca11eafc997
    def get_inputs(self):
        return [
            paddle.uniform([171, 240], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7566cdf98dee30d7ec78c7b1be431163(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d67cbf60ee7a96c433e41952291ebe5a
    def get_inputs(self):
        return [
            paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_d3c3c770492f2c3c04505209f4ae0762(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5a85afe928541aecaaa38de5ecc774b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3c3c770492f2c3c04505209f4ae0762
    def get_inputs(self):
        return [
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_e42c2edb6d5b8f7d7bf167c56d1e8285(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d930f951dac6215d03aedd03ab921ac6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e42c2edb6d5b8f7d7bf167c56d1e8285
    def get_inputs(self):
        return [
            paddle.uniform([1, 512], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_f31afb49bb1f8fd935af7fb6237bfce5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2315e0879db8b2abed4a7f6f4839bccd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f31afb49bb1f8fd935af7fb6237bfce5
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_b2b206736a0d53ca2e46289e74b69b9e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 8, 49, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1f51a7925a9ef443b0f64d8e2007fe9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2b206736a0d53ca2e46289e74b69b9e
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 49, 28, 28], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_79a8dce6a715cf17ef1d0860d1238f5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b187efd56361e4022b806203fc001de9
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([24]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6c7d83a51b4c013ca8ace645bdb22840(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b187efd56361e4022b806203fc001de9
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([24]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0aee7be77787b2994d6e3a2ad186e3fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f42c594134187d886dbe754f9d7c3b5b
    def get_inputs(self):
        return [
            paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c5a0ea36525af911e783d6367e572364(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e796214aa571e77649fd59840cfd70ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3024], dtype='int32'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f2a3361e12b4589918f4793eb5015983(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e641ca14c54a3f9fc8693dce5c5c1b27
    def get_inputs(self):
        return [
            paddle.uniform([1527], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fefe8eaeeacb2076a876d6717760fa9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aff6f421b758d1590f0e3f7805ed276
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024], dtype='int32'), 'bool'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8a0f3df3b848b7b5139570f1e2be86c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d02bec4744d38c009b4a112d792b01a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1527, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8a0f3df3b848b7b5139570f1e2be86c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d02bec4744d38c009b4a112d792b01a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1527, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e10c5edee286c2813eefd62a8ac0f432(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_897cc9a673b22b3e799dea2fcfbb6f08
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_404c6a96bad8c517a2f2e275c3c9f063(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 16, 49, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_29ef53118b986215f90a2add22dbdc75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_404c6a96bad8c517a2f2e275c3c9f063
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9a8adbbc4dbbc9e748a51b207a579fd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b007640545457cf3480b8ca11eafc997
    def get_inputs(self):
        return [
            paddle.uniform([22, 240], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f5dd396ea00b867d2d1180265b75d76f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b187efd56361e4022b806203fc001de9
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0], dtype='int64').reshape([4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9d3469d6d5cb42bed37a91e149740d54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b187efd56361e4022b806203fc001de9
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1, 1, 1], dtype='int64').reshape([4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b0991d71d26094399cd2cc751313b060(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d32ca250ce5245c2f823d77c4185bbb
    def get_inputs(self):
        return [
            paddle.uniform([512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e5567679dfe0c86329e936537f5ccbe6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4cc8b80e88e29763cacea0fbc9e96315
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_36fde7f16d00028a35784a0060621f01(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, 97, 97], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c0cfc614de4862bc3e1c049da41c4c6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36fde7f16d00028a35784a0060621f01
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 97, 97], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_018e7db020f80d63a2e278ccfbd00534(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dea7894d6cc15a393a5335c4b5dbb7e
    def get_inputs(self):
        return [
            paddle.uniform([22, 36], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_29ef53118b986215f90a2add22dbdc75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_404c6a96bad8c517a2f2e275c3c9f063
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b0991d71d26094399cd2cc751313b060(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d32ca250ce5245c2f823d77c4185bbb
    def get_inputs(self):
        return [
            paddle.uniform([512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e5567679dfe0c86329e936537f5ccbe6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4cc8b80e88e29763cacea0fbc9e96315
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5675b37a7e36053f61852749367f9ded(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f42c594134187d886dbe754f9d7c3b5b
    def get_inputs(self):
        return [
            paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ff5e73dd48d93328944b052e53a71146(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2315886efe1966cb990875ac0ccefc79
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 32, 64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ff5e73dd48d93328944b052e53a71146(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2315886efe1966cb990875ac0ccefc79
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 32, 64], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_4a4824a642a34370c6de92bb87b21439(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 672], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1e6dce6dcabda7a28480557172efe27b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a4824a642a34370c6de92bb87b21439
    def get_inputs(self):
        return [
            paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c0bc025a99134e34d661d7698fa49e77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_897cc9a673b22b3e799dea2fcfbb6f08
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_423212f76b26b237aa560b02cb690478(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d67cbf60ee7a96c433e41952291ebe5a
    def get_inputs(self):
        return [
            paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1f51a7925a9ef443b0f64d8e2007fe9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2b206736a0d53ca2e46289e74b69b9e
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 49, 28, 28], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ada06a17739c3ab64e4d1bc6594109fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e796214aa571e77649fd59840cfd70ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bea24e4d7de35cf1de5077d2833c978a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e641ca14c54a3f9fc8693dce5c5c1b27
    def get_inputs(self):
        return [
            paddle.uniform([2066], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fd055d79b0634383a9685e15d444c298(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aff6f421b758d1590f0e3f7805ed276
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_485a24ad7f0830b5c14ee94db1ce87d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d02bec4744d38c009b4a112d792b01a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2066, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_485a24ad7f0830b5c14ee94db1ce87d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d02bec4744d38c009b4a112d792b01a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2066, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e578adf574edb3313f2f50351779f842(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e796214aa571e77649fd59840cfd70ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 9261], dtype='int32'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_00dab0ab3e1f68a57eea62264dd91648(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e641ca14c54a3f9fc8693dce5c5c1b27
    def get_inputs(self):
        return [
            paddle.uniform([4586], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7c44af78bb78cfd0e7436e16fc716673(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aff6f421b758d1590f0e3f7805ed276
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261], dtype='int32'), 'bool'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1d81314bdbdbfc677ca263f96205475c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d02bec4744d38c009b4a112d792b01a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4586, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1d81314bdbdbfc677ca263f96205475c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d02bec4744d38c009b4a112d792b01a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4586, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6291412e5629389e870c1002cd5151df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8e8b8c14d9af9d159aea840edf98b23
    def get_inputs(self):
        return [
            paddle.uniform([6, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e18948509e729db30eef31f1b150d0e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_475cb32d1c927b9d08f8e63628700a91
    def get_inputs(self):
        return [
            paddle.to_tensor([0.21986964344978333, 0.2373102456331253, 0.24102632701396942, 0.2785630226135254, 0.47145286202430725, 0.2085183560848236], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_741b74b13804e198efe614c959dbffa0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aff6f421b758d1590f0e3f7805ed276
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2434], dtype='int32'), 'bool'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6151f203a5fd4c1148162f3b72c2cc25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e796214aa571e77649fd59840cfd70ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5b6b166433161500ef92cde5c90df612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e641ca14c54a3f9fc8693dce5c5c1b27
    def get_inputs(self):
        return [
            paddle.uniform([1073], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_df1e8f45cf629ffd0a3136023843780a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aff6f421b758d1590f0e3f7805ed276
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_10f85683377631bde0148b44472efda0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d02bec4744d38c009b4a112d792b01a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1073, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_10f85683377631bde0148b44472efda0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d02bec4744d38c009b4a112d792b01a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1073, 4], dtype='int64'),
        ]



class PrimitiveOp_c33d4bd298811d807e43fdeb4b58d521(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [0]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_201133578cfa702bb451033393b00c8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c33d4bd298811d807e43fdeb4b58d521
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2726069688796997, 0.024542924016714096, 0.31091731786727905, 0.31940120458602905], dtype='float32').reshape([4]),
        ]



class PrimitiveOp_a55e2e053c2938e1c54781edf25bbedc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_39bd18d1df065b779cbf296968b8bb62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a55e2e053c2938e1c54781edf25bbedc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2726069688796997, 0.024542924016714096, 0.31091731786727905, 0.31940120458602905]], dtype='float32').reshape([1, 4]),
        ]



class PrimitiveOp_64f0403ceeba3e5700f11e53ea100fbc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1b3ade529878086a640cfdf801c9cb60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64f0403ceeba3e5700f11e53ea100fbc
    def get_inputs(self):
        return [
            paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9b293146a4f1a080f2d0657f4d0daebd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65abb7c908c329c7eb1679069b618179
    def get_inputs(self):
        return [
            paddle.uniform([11, 768, 49], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_c1f67abbb5f72d96a714bf601b20bdd8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 32, 49, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1e3414a60c9cd84aa05fa926172ea4d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1f67abbb5f72d96a714bf601b20bdd8
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 49, 7, 7], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b0991d71d26094399cd2cc751313b060(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d32ca250ce5245c2f823d77c4185bbb
    def get_inputs(self):
        return [
            paddle.uniform([512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e5567679dfe0c86329e936537f5ccbe6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4cc8b80e88e29763cacea0fbc9e96315
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_18dd1a8a62014a12c131bb53174fb0cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c33d4bd298811d807e43fdeb4b58d521
    def get_inputs(self):
        return [
            paddle.to_tensor([0.33088600635528564, 0.1025475412607193, 0.42067182064056396, 0.05703792721033096], dtype='float32').reshape([4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_20ec69e5103cd184e38a021189421537(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a55e2e053c2938e1c54781edf25bbedc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.33088600635528564, 0.1025475412607193, 0.42067182064056396, 0.05703792721033096]], dtype='float32').reshape([1, 4]),
        ]



class PrimitiveOp_09bc1054ea761f46a270c1497a0de5bb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1336a7dd7e9617d2643e96d37912c2f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09bc1054ea761f46a270c1497a0de5bb
    def get_inputs(self):
        return [
            paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_bce864fd2bf6f4dddd08e02eef388856(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1248], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7cdb86773d98ef06a83fc4c60d094287(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bce864fd2bf6f4dddd08e02eef388856
    def get_inputs(self):
        return [
            paddle.uniform([1, 1248], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_efa8336c9203233cbf64afe22430fb85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d48269ce2eaabfb5601e3f759c1fa17
    def get_inputs(self):
        return [
            paddle.uniform([171, 480], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_94d695d4b459b0f1a588ecf063277a43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dea7894d6cc15a393a5335c4b5dbb7e
    def get_inputs(self):
        return [
            paddle.uniform([145, 36], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_55ada0550b7ce2163824ef3559f42708(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64742105392b863a7fe416d40d6b56c8
    def get_inputs(self):
        return [
            paddle.uniform([15, 25], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_55ada0550b7ce2163824ef3559f42708(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64742105392b863a7fe416d40d6b56c8
    def get_inputs(self):
        return [
            paddle.uniform([15, 25], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7b5d3cb963937780ea285463213e9af1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2315886efe1966cb990875ac0ccefc79
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 128, 256], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7b5d3cb963937780ea285463213e9af1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2315886efe1966cb990875ac0ccefc79
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 128, 256], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_229e231aae14080faae44e111df71aea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e796214aa571e77649fd59840cfd70ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4725], dtype='int32'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4bf9bf0ba4a9e20bbf52a425a0cd5f5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e641ca14c54a3f9fc8693dce5c5c1b27
    def get_inputs(self):
        return [
            paddle.uniform([2383], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_29d79aa9f8b0c4092a405d8f9201eafd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aff6f421b758d1590f0e3f7805ed276
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725], dtype='int32'), 'bool'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_df9a72f5516630d0a21c5060e09143a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d02bec4744d38c009b4a112d792b01a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2383, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_df9a72f5516630d0a21c5060e09143a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d02bec4744d38c009b4a112d792b01a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2383, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0a080d0b143c7453e4502672db6e98e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2315886efe1966cb990875ac0ccefc79
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 32, 64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0a080d0b143c7453e4502672db6e98e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2315886efe1966cb990875ac0ccefc79
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 32, 64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1b0110230cff7fae95e6efcf42fe71d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e796214aa571e77649fd59840cfd70ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 6069], dtype='int32'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a53212b7e15c28ed66ed6ab5bda25115(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e641ca14c54a3f9fc8693dce5c5c1b27
    def get_inputs(self):
        return [
            paddle.uniform([3030], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fcab399e5d93a97b7eefefac118a2f54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aff6f421b758d1590f0e3f7805ed276
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069], dtype='int32'), 'bool'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5dc079b215846df5d76f7a33a046704a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d02bec4744d38c009b4a112d792b01a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3030, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5dc079b215846df5d76f7a33a046704a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d02bec4744d38c009b4a112d792b01a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3030, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1bb44e9012304883db90cfdc69b48f70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e796214aa571e77649fd59840cfd70ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 7581], dtype='int32'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_19d9bfe671f93a67cdf4fbfbb0ab1599(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e641ca14c54a3f9fc8693dce5c5c1b27
    def get_inputs(self):
        return [
            paddle.uniform([3787], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c56c9e7caf434be0c772e0821cff7f2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aff6f421b758d1590f0e3f7805ed276
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581], dtype='int32'), 'bool'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cb01b42e62748bc4c3be2f18a61d76eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d02bec4744d38c009b4a112d792b01a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3787, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cb01b42e62748bc4c3be2f18a61d76eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d02bec4744d38c009b4a112d792b01a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3787, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_55ec174a9365e42b488046007688e9c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2315886efe1966cb990875ac0ccefc79
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 128, 256], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_55ec174a9365e42b488046007688e9c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2315886efe1966cb990875ac0ccefc79
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 128, 256], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_bd6e11a231e4f10681ff19c951ee4917(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 156], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_eac70dea019657ef3ffb93399a24ed30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd6e11a231e4f10681ff19c951ee4917
    def get_inputs(self):
        return [
            paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3c128f36f7cdb4ed61184ace32749e40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2315886efe1966cb990875ac0ccefc79
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 64, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3c128f36f7cdb4ed61184ace32749e40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2315886efe1966cb990875ac0ccefc79
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 64, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_afe9f2f9e806ecf43943f8f7e30d0bef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee828dc99c1f501282115dbef8bec917
    def get_inputs(self):
        return [
            paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_e9c87a2b6edd2a0fa1d29d627064c9af(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 8, 49, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_11b3aaa55f765d6f0735dfd9cbebaaef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9c87a2b6edd2a0fa1d29d627064c9af
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 49, 28, 28], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_11f9709d8b4cb0ba133be332b79455bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d48269ce2eaabfb5601e3f759c1fa17
    def get_inputs(self):
        return [
            paddle.uniform([22, 480], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_21ecfe293766747e5145ee4db75b825d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d48269ce2eaabfb5601e3f759c1fa17
    def get_inputs(self):
        return [
            paddle.uniform([145, 480], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_98e6f3b242707a28e86e5c816376080f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8e8b8c14d9af9d159aea840edf98b23
    def get_inputs(self):
        return [
            paddle.uniform([2, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e02a3ad24aca97b8fe77f00d1940ddd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_475cb32d1c927b9d08f8e63628700a91
    def get_inputs(self):
        return [
            paddle.to_tensor([0.37492507696151733, 0.31706711649894714], dtype='float32').reshape([2]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3b50b3080e66da2db273bb8bf10f4243(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dea7894d6cc15a393a5335c4b5dbb7e
    def get_inputs(self):
        return [
            paddle.uniform([171, 36], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_5137007da27ed68d1f139b2cb8c7f4b9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 16, 49, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c153f36da2bc690073731fea098a4cd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5137007da27ed68d1f139b2cb8c7f4b9
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_8d547845c3d35eba610b798c5888852e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 120], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_393573b2bd0ef33d3fe655cddc499f88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d547845c3d35eba610b798c5888852e
    def get_inputs(self):
        return [
            paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8ff67d519e5fbeba0c3c00a573ef9f28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb80a75edd5f8907b3cdb9307144b45
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 49, 7, 7], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_922a1eaa6011ada12d686640123ca858(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b187efd56361e4022b806203fc001de9
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([20]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e0cdd43a935f34adc739ffcab45efb34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b187efd56361e4022b806203fc001de9
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([20]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1e6dce6dcabda7a28480557172efe27b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a4824a642a34370c6de92bb87b21439
    def get_inputs(self):
        return [
            paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_11b3aaa55f765d6f0735dfd9cbebaaef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9c87a2b6edd2a0fa1d29d627064c9af
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 49, 28, 28], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ada06a17739c3ab64e4d1bc6594109fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e796214aa571e77649fd59840cfd70ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3efd63bc2c9b20350959022953beffb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e641ca14c54a3f9fc8693dce5c5c1b27
    def get_inputs(self):
        return [
            paddle.uniform([2084], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fd055d79b0634383a9685e15d444c298(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aff6f421b758d1590f0e3f7805ed276
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_802b96fff9ede2c7560e8ff9bf48624e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d02bec4744d38c009b4a112d792b01a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2084, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_802b96fff9ede2c7560e8ff9bf48624e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d02bec4744d38c009b4a112d792b01a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2084, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_28eb6c514b1336191185c77482b988fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77dcac572ff7d24b9e7ddf7971c603e8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 500], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b139dca4b759021550f9c05fd13dae78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ea9493df868c15c17fe1346efc13bbb
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 500], dtype='int32'),
        ]



class PrimitiveOp_04212e5b64899859301a2ec440c4a1be(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 2, 9, 112, 112], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3fff71f92f407b8d7ab3a8ba638e1716(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04212e5b64899859301a2ec440c4a1be
    def get_inputs(self):
        return [
            paddle.uniform([22, 2, 9, 112, 112], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1e3414a60c9cd84aa05fa926172ea4d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1f67abbb5f72d96a714bf601b20bdd8
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 49, 7, 7], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b4fb38bc8d8dbb8fccc671db2beb63d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd329f5fde19fcac72c8c3ae628c0049
    def get_inputs(self):
        return [
            paddle.uniform([150, 256], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f4896d395c2c3c80e136ff0897b51294(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e796214aa571e77649fd59840cfd70ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 8400], dtype='int32'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0982522fb3c3df0d8a2ae63b11316498(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e641ca14c54a3f9fc8693dce5c5c1b27
    def get_inputs(self):
        return [
            paddle.uniform([4260], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_537922de7c40f485b7b5f158b2cf8f51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aff6f421b758d1590f0e3f7805ed276
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400], dtype='int32'), 'bool'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1e8647852f8b8b13beffd81e64edb270(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d02bec4744d38c009b4a112d792b01a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4260, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1e8647852f8b8b13beffd81e64edb270(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d02bec4744d38c009b4a112d792b01a
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4260, 4], dtype='int64'),
        ]



class PrimitiveOp_3e3996b0d303c1bfca2eed0fbc11feb2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 624], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_191f0ccbb127285e75d591898a1ac03e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e3996b0d303c1bfca2eed0fbc11feb2
    def get_inputs(self):
        return [
            paddle.uniform([1, 624], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c153f36da2bc690073731fea098a4cd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5137007da27ed68d1f139b2cb8c7f4b9
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_298a3c3df4c2bd7166d5812d103bfaa5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64742105392b863a7fe416d40d6b56c8
    def get_inputs(self):
        return [
            paddle.uniform([24, 36], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_298a3c3df4c2bd7166d5812d103bfaa5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64742105392b863a7fe416d40d6b56c8
    def get_inputs(self):
        return [
            paddle.uniform([24, 36], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_36758dc630f24d8f48fe58c2546f5f5f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2, 3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8b3c774c2ee2a1f4b4278a34c22b11e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36758dc630f24d8f48fe58c2546f5f5f
    def get_inputs(self):
        return [
            paddle.uniform([1, 72], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_47d7720314a2d9a721aa9dd0e8b16ced(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36758dc630f24d8f48fe58c2546f5f5f
    def get_inputs(self):
        return [
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e6db66d1db195866adedf3b9b1a9357d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36758dc630f24d8f48fe58c2546f5f5f
    def get_inputs(self):
        return [
            paddle.uniform([1, 960], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_74e8d9e6afa79740278132c4fbfe8112(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36758dc630f24d8f48fe58c2546f5f5f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_480bbf02648c3a19dddbff1e80a4b9ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36758dc630f24d8f48fe58c2546f5f5f
    def get_inputs(self):
        return [
            paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_104b8d70cccc1230e258ef3770fb658c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36758dc630f24d8f48fe58c2546f5f5f
    def get_inputs(self):
        return [
            paddle.uniform([4, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_59613c9d825c01ba2fb468a520b1b52e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_475cb32d1c927b9d08f8e63628700a91
    def get_inputs(self):
        return [
            paddle.to_tensor([0.33804893493652344, 0.39942264556884766, 0.09662508219480515, 0.1508401781320572], dtype='float32').reshape([4]),
        ]



class PrimitiveOp_45635ae50ce12c463fd25094d5307ceb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b06fd859e0d647753580bf30ae5dc3aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45635ae50ce12c463fd25094d5307ceb
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 9, 112, 112], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4c8149e3286a7df3e8d7e6ffa3a7ca12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_897cc9a673b22b3e799dea2fcfbb6f08
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_28eb6c514b1336191185c77482b988fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77dcac572ff7d24b9e7ddf7971c603e8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 500], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b139dca4b759021550f9c05fd13dae78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ea9493df868c15c17fe1346efc13bbb
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 500], dtype='int32'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4ff44736cb61bac3026ead8da0b7847c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36758dc630f24d8f48fe58c2546f5f5f
    def get_inputs(self):
        return [
            paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d21e1fe8c6c3aa891b82e1359054f576(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aff6f421b758d1590f0e3f7805ed276
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8732], dtype='int32'), 'bool'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1a330eb41b440b5abb61c2b8cb2678ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36758dc630f24d8f48fe58c2546f5f5f
    def get_inputs(self):
        return [
            paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_51651f4c48c5a43e815e355416c199b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd329f5fde19fcac72c8c3ae628c0049
    def get_inputs(self):
        return [
            paddle.uniform([21, 256], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_28eb6c514b1336191185c77482b988fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77dcac572ff7d24b9e7ddf7971c603e8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 500], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b139dca4b759021550f9c05fd13dae78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ea9493df868c15c17fe1346efc13bbb
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 500], dtype='int32'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1a330eb41b440b5abb61c2b8cb2678ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36758dc630f24d8f48fe58c2546f5f5f
    def get_inputs(self):
        return [
            paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5993e87c50c296a408bdc04991c44233(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d32ca250ce5245c2f823d77c4185bbb
    def get_inputs(self):
        return [
            paddle.uniform([1, 21824, 2], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_02ad572044a2cfb4b003d96bd41a603f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f0d6bc32e9943d2d3694cb97215d4c4
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.29133909940719604, 0.4853375554084778], [0.36761757731437683, 0.4571003019809723], [0.46794551610946655, 0.1688530147075653], [0.2670198976993561, 0.1982056200504303], [0.12026780098676682, 0.48663008213043213], [0.2773101329803467, 0.48897823691368103]]], dtype='float32').reshape([1, 6, 2]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5a7159266ec46ca019eee0d649ed20fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45635ae50ce12c463fd25094d5307ceb
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 49, 56, 56], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1a1be7b66172e1787d2855997c39f2ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b187efd56361e4022b806203fc001de9
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([16]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ce05199e78ede71a9dca2db88cb9d450(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b187efd56361e4022b806203fc001de9
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([16]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cafbbab16d345f19d77b202915f84b68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36758dc630f24d8f48fe58c2546f5f5f
    def get_inputs(self):
        return [
            paddle.uniform([145, 240], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_eebff72108b454f9e0a444f2c563b1bf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_23a2970342884f4d46e21865ec1e71f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eebff72108b454f9e0a444f2c563b1bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_52c0e6d78bc6f145b0b882982799801c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd329f5fde19fcac72c8c3ae628c0049
    def get_inputs(self):
        return [
            paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3cb823eb7dfa307fecdec8329746ff1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de692023e6733e944677e99243258156
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.18305830657482147, 0.25759702920913696, 0.46923521161079407, 0.37422502040863037]], dtype='float32').reshape([1, 4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fb2d904af97db25eb9faf8cb05ddfb36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd329f5fde19fcac72c8c3ae628c0049
    def get_inputs(self):
        return [
            paddle.uniform([300, 256], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_63eba50cf16fdf1050dc069ab09caf3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2315886efe1966cb990875ac0ccefc79
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 64, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_63eba50cf16fdf1050dc069ab09caf3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2315886efe1966cb990875ac0ccefc79
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 64, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b0991d71d26094399cd2cc751313b060(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d32ca250ce5245c2f823d77c4185bbb
    def get_inputs(self):
        return [
            paddle.uniform([512, 64, 128], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_3b9dea649e1064048151917046b5455f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7d0cae73af376f13b7ebb4de76c459e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b9dea649e1064048151917046b5455f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fd132e51864e27315836cdba6aa9771e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64742105392b863a7fe416d40d6b56c8
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fd132e51864e27315836cdba6aa9771e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64742105392b863a7fe416d40d6b56c8
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ca408fb71de0097365dd1cd86bea967f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eebff72108b454f9e0a444f2c563b1bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 21], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fd967e7d4fca3e79eec7114031751df5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36758dc630f24d8f48fe58c2546f5f5f
    def get_inputs(self):
        return [
            paddle.uniform([3, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c63f20bf2174b4e82ae2ac12a1b35723(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_475cb32d1c927b9d08f8e63628700a91
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2647590637207031, 0.05532848462462425, 0.46784508228302], dtype='float32').reshape([3]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6b0d414e4915b28840b43cf4e2b4abce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36758dc630f24d8f48fe58c2546f5f5f
    def get_inputs(self):
        return [
            paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_662ec17da66202aa063b37d8fd9d2718(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36758dc630f24d8f48fe58c2546f5f5f
    def get_inputs(self):
        return [
            paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4cb59ce08bafe761551caf2cb3e3c893(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e796214aa571e77649fd59840cfd70ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_73676d7b6b3608fc76c99a5ab8883279(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e641ca14c54a3f9fc8693dce5c5c1b27
    def get_inputs(self):
        return [
            paddle.uniform([1777], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_eea885eda6722405ec5b3fe2cd6ca875(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aff6f421b758d1590f0e3f7805ed276
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
        ]



class PrimitiveOp_b4e5503d7644584b8b5cd6a103349c95(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_324d278b8bca5352cc80a0cb9e63cec7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4e5503d7644584b8b5cd6a103349c95
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1777, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_324d278b8bca5352cc80a0cb9e63cec7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4e5503d7644584b8b5cd6a103349c95
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1777, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d1ba08425efedfd97862a878d318558d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64742105392b863a7fe416d40d6b56c8
    def get_inputs(self):
        return [
            paddle.uniform([20, 30], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d1ba08425efedfd97862a878d318558d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64742105392b863a7fe416d40d6b56c8
    def get_inputs(self):
        return [
            paddle.uniform([20, 30], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3e0414912a0301cc0b43915aac3b4793(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45635ae50ce12c463fd25094d5307ceb
    def get_inputs(self):
        return [
            paddle.uniform([22, 4, 49, 56, 56], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d2da0ff46c7c63dc215bbac6ea17785c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36758dc630f24d8f48fe58c2546f5f5f
    def get_inputs(self):
        return [
            paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ebc7c3721ef4860a8c25dba6ffa5203d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f0d6bc32e9943d2d3694cb97215d4c4
    def get_inputs(self):
        return [
            paddle.uniform([43, 768, 49], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0b4349c07fca865c92b7c4f61e698535(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36758dc630f24d8f48fe58c2546f5f5f
    def get_inputs(self):
        return [
            paddle.uniform([10, 240], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3866202279d18be06b70d7021539861f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e796214aa571e77649fd59840cfd70ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 11109], dtype='int32'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bf224eec18902dcb00eeb4ab5a04afd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e641ca14c54a3f9fc8693dce5c5c1b27
    def get_inputs(self):
        return [
            paddle.uniform([5480], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b4133880239758627b547f5798c630b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aff6f421b758d1590f0e3f7805ed276
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109], dtype='int32'), 'bool'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ae2e7ab94d6610f17311eedf6dd5d8de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4e5503d7644584b8b5cd6a103349c95
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[5480, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ae2e7ab94d6610f17311eedf6dd5d8de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4e5503d7644584b8b5cd6a103349c95
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[5480, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_81859d329c659b73fd9a030e4a33d212(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b187efd56361e4022b806203fc001de9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[36], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_81859d329c659b73fd9a030e4a33d212(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b187efd56361e4022b806203fc001de9
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[36], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_480bbf02648c3a19dddbff1e80a4b9ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36758dc630f24d8f48fe58c2546f5f5f
    def get_inputs(self):
        return [
            paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ff0617d823b50f8cf2909ba31d6e3d89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45635ae50ce12c463fd25094d5307ceb
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 49, 7, 7], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6dc24f4cb87d059a221da5164e3c82f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd329f5fde19fcac72c8c3ae628c0049
    def get_inputs(self):
        return [
            paddle.uniform([19, 256], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_498893042295ce3d276667f8c6086b89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36758dc630f24d8f48fe58c2546f5f5f
    def get_inputs(self):
        return [
            paddle.uniform([10, 36], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_06f01b0023f49421d8dcd22ec7e25759(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36758dc630f24d8f48fe58c2546f5f5f
    def get_inputs(self):
        return [
            paddle.uniform([10, 480], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4cb59ce08bafe761551caf2cb3e3c893(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e796214aa571e77649fd59840cfd70ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2ddf54be1dddfaa8d15d815adb5f9a23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e641ca14c54a3f9fc8693dce5c5c1b27
    def get_inputs(self):
        return [
            paddle.uniform([1742], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_eea885eda6722405ec5b3fe2cd6ca875(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aff6f421b758d1590f0e3f7805ed276
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7c02221bd0857e7f658e24704eba0c7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4e5503d7644584b8b5cd6a103349c95
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1742, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7c02221bd0857e7f658e24704eba0c7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4e5503d7644584b8b5cd6a103349c95
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1742, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b2c336b4264e79407eabbb9397629ea9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36758dc630f24d8f48fe58c2546f5f5f
    def get_inputs(self):
        return [
            paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_286943a88db1b5eb5134e095878660ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36758dc630f24d8f48fe58c2546f5f5f
    def get_inputs(self):
        return [
            paddle.uniform([171, 240], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d2da0ff46c7c63dc215bbac6ea17785c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36758dc630f24d8f48fe58c2546f5f5f
    def get_inputs(self):
        return [
            paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1ba12c27bf4ed1045b331e6300765867(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c33d4bd298811d807e43fdeb4b58d521
    def get_inputs(self):
        return [
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_0f223e915be4f0be650b9bd177b9ba3c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6f9d7d347197ad575386e6aee5172761(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f223e915be4f0be650b9bd177b9ba3c
    def get_inputs(self):
        return [
            paddle.uniform([1, 512], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_74e45202a301c29f03df10971102c057(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [3]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9429a070b93e60a1cba734cfca071847(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74e45202a301c29f03df10971102c057
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_55ace8dc9c9b2887f7c87c3a1aaec56a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45635ae50ce12c463fd25094d5307ceb
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 49, 28, 28], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_79a8dce6a715cf17ef1d0860d1238f5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b187efd56361e4022b806203fc001de9
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([24]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6c7d83a51b4c013ca8ace645bdb22840(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b187efd56361e4022b806203fc001de9
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([24]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_46386f56212ccb801e53ca9b8110e5a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36758dc630f24d8f48fe58c2546f5f5f
    def get_inputs(self):
        return [
            paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c5a0ea36525af911e783d6367e572364(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e796214aa571e77649fd59840cfd70ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 3024], dtype='int32'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f2a3361e12b4589918f4793eb5015983(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e641ca14c54a3f9fc8693dce5c5c1b27
    def get_inputs(self):
        return [
            paddle.uniform([1527], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fefe8eaeeacb2076a876d6717760fa9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aff6f421b758d1590f0e3f7805ed276
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024], dtype='int32'), 'bool'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_09328aebf1895f2a951b79ef4aaf0bc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4e5503d7644584b8b5cd6a103349c95
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1527, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_09328aebf1895f2a951b79ef4aaf0bc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4e5503d7644584b8b5cd6a103349c95
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1527, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e10c5edee286c2813eefd62a8ac0f432(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_897cc9a673b22b3e799dea2fcfbb6f08
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_19a5c03d8554f2ba3a59d8e2b92bd315(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45635ae50ce12c463fd25094d5307ceb
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8362d38630597e130a0b5c9b59ad10ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36758dc630f24d8f48fe58c2546f5f5f
    def get_inputs(self):
        return [
            paddle.uniform([22, 240], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f5dd396ea00b867d2d1180265b75d76f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b187efd56361e4022b806203fc001de9
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0], dtype='int64').reshape([4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9d3469d6d5cb42bed37a91e149740d54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b187efd56361e4022b806203fc001de9
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1, 1, 1], dtype='int64').reshape([4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b0991d71d26094399cd2cc751313b060(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d32ca250ce5245c2f823d77c4185bbb
    def get_inputs(self):
        return [
            paddle.uniform([512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7d0cae73af376f13b7ebb4de76c459e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b9dea649e1064048151917046b5455f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c46a7a191ddbcbcca3b766851d4b16a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b9dea649e1064048151917046b5455f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 97, 97], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a00e226cd4a9ff1988388572ceaba9c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36758dc630f24d8f48fe58c2546f5f5f
    def get_inputs(self):
        return [
            paddle.uniform([22, 36], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_19a5c03d8554f2ba3a59d8e2b92bd315(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45635ae50ce12c463fd25094d5307ceb
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b0991d71d26094399cd2cc751313b060(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d32ca250ce5245c2f823d77c4185bbb
    def get_inputs(self):
        return [
            paddle.uniform([512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7d0cae73af376f13b7ebb4de76c459e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b9dea649e1064048151917046b5455f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_59f665e141e6413c0b7ec1c6893e35d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36758dc630f24d8f48fe58c2546f5f5f
    def get_inputs(self):
        return [
            paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ff5e73dd48d93328944b052e53a71146(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2315886efe1966cb990875ac0ccefc79
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 32, 64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ff5e73dd48d93328944b052e53a71146(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2315886efe1966cb990875ac0ccefc79
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 32, 64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f3917a5d06c3d2b9dd3d5b368ed8105f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36758dc630f24d8f48fe58c2546f5f5f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c0bc025a99134e34d661d7698fa49e77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_897cc9a673b22b3e799dea2fcfbb6f08
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b2c336b4264e79407eabbb9397629ea9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36758dc630f24d8f48fe58c2546f5f5f
    def get_inputs(self):
        return [
            paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_55ace8dc9c9b2887f7c87c3a1aaec56a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45635ae50ce12c463fd25094d5307ceb
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 49, 28, 28], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ada06a17739c3ab64e4d1bc6594109fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e796214aa571e77649fd59840cfd70ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bea24e4d7de35cf1de5077d2833c978a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e641ca14c54a3f9fc8693dce5c5c1b27
    def get_inputs(self):
        return [
            paddle.uniform([2066], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fd055d79b0634383a9685e15d444c298(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aff6f421b758d1590f0e3f7805ed276
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3b90e8d30a99c95cfed77185ee5f624c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4e5503d7644584b8b5cd6a103349c95
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2066, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3b90e8d30a99c95cfed77185ee5f624c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4e5503d7644584b8b5cd6a103349c95
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2066, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e578adf574edb3313f2f50351779f842(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e796214aa571e77649fd59840cfd70ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 9261], dtype='int32'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_00dab0ab3e1f68a57eea62264dd91648(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e641ca14c54a3f9fc8693dce5c5c1b27
    def get_inputs(self):
        return [
            paddle.uniform([4586], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7c44af78bb78cfd0e7436e16fc716673(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aff6f421b758d1590f0e3f7805ed276
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261], dtype='int32'), 'bool'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0353c5e33ed7894a8666d075c69a99ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4e5503d7644584b8b5cd6a103349c95
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4586, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0353c5e33ed7894a8666d075c69a99ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4e5503d7644584b8b5cd6a103349c95
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4586, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ba62a67d84a78e76e2c792a245173216(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36758dc630f24d8f48fe58c2546f5f5f
    def get_inputs(self):
        return [
            paddle.uniform([6, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e18948509e729db30eef31f1b150d0e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_475cb32d1c927b9d08f8e63628700a91
    def get_inputs(self):
        return [
            paddle.to_tensor([0.21986964344978333, 0.2373102456331253, 0.24102632701396942, 0.2785630226135254, 0.47145286202430725, 0.2085183560848236], dtype='float32').reshape([6]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_741b74b13804e198efe614c959dbffa0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aff6f421b758d1590f0e3f7805ed276
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2434], dtype='int32'), 'bool'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6151f203a5fd4c1148162f3b72c2cc25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e796214aa571e77649fd59840cfd70ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5b6b166433161500ef92cde5c90df612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e641ca14c54a3f9fc8693dce5c5c1b27
    def get_inputs(self):
        return [
            paddle.uniform([1073], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_df1e8f45cf629ffd0a3136023843780a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aff6f421b758d1590f0e3f7805ed276
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_db521d0327368f3f12bc444da5f1241f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4e5503d7644584b8b5cd6a103349c95
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1073, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_db521d0327368f3f12bc444da5f1241f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4e5503d7644584b8b5cd6a103349c95
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1073, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_201133578cfa702bb451033393b00c8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c33d4bd298811d807e43fdeb4b58d521
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2726069688796997, 0.024542924016714096, 0.31091731786727905, 0.31940120458602905], dtype='float32').reshape([4]),
        ]



class PrimitiveOp_a04262ccd1e15fd15df41f0e0e181301(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        return paddle.unsqueeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d664843cb7100c428227fda567c64d4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a04262ccd1e15fd15df41f0e0e181301
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2726069688796997, 0.024542924016714096, 0.31091731786727905, 0.31940120458602905]], dtype='float32').reshape([1, 4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7c15bbb9ca10da95150aa87ce490fc5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de692023e6733e944677e99243258156
    def get_inputs(self):
        return [
            paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_de8eccb268baaf5d1507d352d44c7f35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f0d6bc32e9943d2d3694cb97215d4c4
    def get_inputs(self):
        return [
            paddle.uniform([11, 768, 49], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_05591d175f3007f23db469833206da8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45635ae50ce12c463fd25094d5307ceb
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 49, 7, 7], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b0991d71d26094399cd2cc751313b060(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d32ca250ce5245c2f823d77c4185bbb
    def get_inputs(self):
        return [
            paddle.uniform([512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7d0cae73af376f13b7ebb4de76c459e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b9dea649e1064048151917046b5455f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_18dd1a8a62014a12c131bb53174fb0cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c33d4bd298811d807e43fdeb4b58d521
    def get_inputs(self):
        return [
            paddle.to_tensor([0.33088600635528564, 0.1025475412607193, 0.42067182064056396, 0.05703792721033096], dtype='float32').reshape([4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c35a323063dce6099668560171d60ecd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a04262ccd1e15fd15df41f0e0e181301
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.33088600635528564, 0.1025475412607193, 0.42067182064056396, 0.05703792721033096]], dtype='float32').reshape([1, 4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f1626bfb6fed254bb28822538dec0cc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de692023e6733e944677e99243258156
    def get_inputs(self):
        return [
            paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f1ac1b4697b1b481793fbe2f133369e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36758dc630f24d8f48fe58c2546f5f5f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1248], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_192ce88553ffb8fca233c1562a41d886(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36758dc630f24d8f48fe58c2546f5f5f
    def get_inputs(self):
        return [
            paddle.uniform([171, 480], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_50d85f2b9b57751331a745b141fa0271(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36758dc630f24d8f48fe58c2546f5f5f
    def get_inputs(self):
        return [
            paddle.uniform([145, 36], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_55ada0550b7ce2163824ef3559f42708(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64742105392b863a7fe416d40d6b56c8
    def get_inputs(self):
        return [
            paddle.uniform([15, 25], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_55ada0550b7ce2163824ef3559f42708(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64742105392b863a7fe416d40d6b56c8
    def get_inputs(self):
        return [
            paddle.uniform([15, 25], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7b5d3cb963937780ea285463213e9af1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2315886efe1966cb990875ac0ccefc79
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 128, 256], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7b5d3cb963937780ea285463213e9af1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2315886efe1966cb990875ac0ccefc79
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 128, 256], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_229e231aae14080faae44e111df71aea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e796214aa571e77649fd59840cfd70ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4725], dtype='int32'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4bf9bf0ba4a9e20bbf52a425a0cd5f5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e641ca14c54a3f9fc8693dce5c5c1b27
    def get_inputs(self):
        return [
            paddle.uniform([2383], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_29d79aa9f8b0c4092a405d8f9201eafd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aff6f421b758d1590f0e3f7805ed276
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725], dtype='int32'), 'bool'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_37ee5624e5058b5eaba020e3fe9a1672(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4e5503d7644584b8b5cd6a103349c95
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2383, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_37ee5624e5058b5eaba020e3fe9a1672(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4e5503d7644584b8b5cd6a103349c95
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2383, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0a080d0b143c7453e4502672db6e98e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2315886efe1966cb990875ac0ccefc79
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 32, 64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0a080d0b143c7453e4502672db6e98e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2315886efe1966cb990875ac0ccefc79
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 32, 64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1b0110230cff7fae95e6efcf42fe71d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e796214aa571e77649fd59840cfd70ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 6069], dtype='int32'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a53212b7e15c28ed66ed6ab5bda25115(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e641ca14c54a3f9fc8693dce5c5c1b27
    def get_inputs(self):
        return [
            paddle.uniform([3030], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fcab399e5d93a97b7eefefac118a2f54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aff6f421b758d1590f0e3f7805ed276
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069], dtype='int32'), 'bool'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fc398f10ed36a998096359c396dde993(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4e5503d7644584b8b5cd6a103349c95
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3030, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fc398f10ed36a998096359c396dde993(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4e5503d7644584b8b5cd6a103349c95
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3030, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1bb44e9012304883db90cfdc69b48f70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e796214aa571e77649fd59840cfd70ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 7581], dtype='int32'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_19d9bfe671f93a67cdf4fbfbb0ab1599(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e641ca14c54a3f9fc8693dce5c5c1b27
    def get_inputs(self):
        return [
            paddle.uniform([3787], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c56c9e7caf434be0c772e0821cff7f2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aff6f421b758d1590f0e3f7805ed276
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581], dtype='int32'), 'bool'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7b3b7990575320810487d89960a460bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4e5503d7644584b8b5cd6a103349c95
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3787, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7b3b7990575320810487d89960a460bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4e5503d7644584b8b5cd6a103349c95
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3787, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_55ec174a9365e42b488046007688e9c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2315886efe1966cb990875ac0ccefc79
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 128, 256], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_55ec174a9365e42b488046007688e9c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2315886efe1966cb990875ac0ccefc79
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 128, 256], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_457a606f39a66a82d2e4c21669f71b71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36758dc630f24d8f48fe58c2546f5f5f
    def get_inputs(self):
        return [
            paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3c128f36f7cdb4ed61184ace32749e40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2315886efe1966cb990875ac0ccefc79
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 64, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3c128f36f7cdb4ed61184ace32749e40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2315886efe1966cb990875ac0ccefc79
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 64, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_662ec17da66202aa063b37d8fd9d2718(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36758dc630f24d8f48fe58c2546f5f5f
    def get_inputs(self):
        return [
            paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6a99e07948fa26de29c9b8fe08f7015c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45635ae50ce12c463fd25094d5307ceb
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 49, 28, 28], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e5942376b0aecbf78ec3f0ee3a3e654b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36758dc630f24d8f48fe58c2546f5f5f
    def get_inputs(self):
        return [
            paddle.uniform([22, 480], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_229980c169db6b5a28485e1fac33f9dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36758dc630f24d8f48fe58c2546f5f5f
    def get_inputs(self):
        return [
            paddle.uniform([145, 480], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d22df89e48fda79f09ef3d1a70c64cb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36758dc630f24d8f48fe58c2546f5f5f
    def get_inputs(self):
        return [
            paddle.uniform([2, 80], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e02a3ad24aca97b8fe77f00d1940ddd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_475cb32d1c927b9d08f8e63628700a91
    def get_inputs(self):
        return [
            paddle.to_tensor([0.37492507696151733, 0.31706711649894714], dtype='float32').reshape([2]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_54f2f43b2541345926d39a22274ea124(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36758dc630f24d8f48fe58c2546f5f5f
    def get_inputs(self):
        return [
            paddle.uniform([171, 36], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_746569aee72f83c47fc79185b1e73995(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45635ae50ce12c463fd25094d5307ceb
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_244c50afea281b0fbb9b6ae85087d14c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36758dc630f24d8f48fe58c2546f5f5f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ff0617d823b50f8cf2909ba31d6e3d89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45635ae50ce12c463fd25094d5307ceb
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 49, 7, 7], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_922a1eaa6011ada12d686640123ca858(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b187efd56361e4022b806203fc001de9
    def get_inputs(self):
        return [
            paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([20]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e0cdd43a935f34adc739ffcab45efb34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b187efd56361e4022b806203fc001de9
    def get_inputs(self):
        return [
            paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([20]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f3917a5d06c3d2b9dd3d5b368ed8105f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36758dc630f24d8f48fe58c2546f5f5f
    def get_inputs(self):
        return [
            paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6a99e07948fa26de29c9b8fe08f7015c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45635ae50ce12c463fd25094d5307ceb
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 49, 28, 28], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ada06a17739c3ab64e4d1bc6594109fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e796214aa571e77649fd59840cfd70ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3efd63bc2c9b20350959022953beffb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e641ca14c54a3f9fc8693dce5c5c1b27
    def get_inputs(self):
        return [
            paddle.uniform([2084], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fd055d79b0634383a9685e15d444c298(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aff6f421b758d1590f0e3f7805ed276
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8e49c31958bbc20946c40559da8dc204(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4e5503d7644584b8b5cd6a103349c95
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2084, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8e49c31958bbc20946c40559da8dc204(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4e5503d7644584b8b5cd6a103349c95
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2084, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_28eb6c514b1336191185c77482b988fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77dcac572ff7d24b9e7ddf7971c603e8
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 500], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b139dca4b759021550f9c05fd13dae78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ea9493df868c15c17fe1346efc13bbb
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 500], dtype='int32'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a0ae79015b812f995cf049e261181abd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45635ae50ce12c463fd25094d5307ceb
    def get_inputs(self):
        return [
            paddle.uniform([22, 2, 9, 112, 112], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_05591d175f3007f23db469833206da8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45635ae50ce12c463fd25094d5307ceb
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 49, 7, 7], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b4fb38bc8d8dbb8fccc671db2beb63d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd329f5fde19fcac72c8c3ae628c0049
    def get_inputs(self):
        return [
            paddle.uniform([150, 256], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f4896d395c2c3c80e136ff0897b51294(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e796214aa571e77649fd59840cfd70ea
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 8400], dtype='int32'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0982522fb3c3df0d8a2ae63b11316498(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e641ca14c54a3f9fc8693dce5c5c1b27
    def get_inputs(self):
        return [
            paddle.uniform([4260], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_537922de7c40f485b7b5f158b2cf8f51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aff6f421b758d1590f0e3f7805ed276
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400], dtype='int32'), 'bool'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a44834ca997ace5fc71ac0ae017ba671(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4e5503d7644584b8b5cd6a103349c95
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4260, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a44834ca997ace5fc71ac0ae017ba671(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4e5503d7644584b8b5cd6a103349c95
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[4260, 4], dtype='int64'),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e490e24eda527c4b281749277d4c2e99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36758dc630f24d8f48fe58c2546f5f5f
    def get_inputs(self):
        return [
            paddle.uniform([1, 624], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_746569aee72f83c47fc79185b1e73995(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45635ae50ce12c463fd25094d5307ceb
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()