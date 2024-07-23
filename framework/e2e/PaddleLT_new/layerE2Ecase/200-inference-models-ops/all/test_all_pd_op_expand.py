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
import itertools

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

def GetExitCodeAndStdErr(cmd, env):
    env = {
        k:v
        for k, v in env.items()
        if v is not None
    }
    import subprocess
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    return result.returncode, result.stderr

def GetStageExitCodeAndStdErr(stage):
    return GetExitCodeAndStdErr(
        [sys.executable, __file__],
        env=dict(
            PADDLE_DEBUG_CINN_STAGE_NAME=stage.name,
            PADDLE_DEBUG_CINN_STAGE_ENABLE_DIFF='0',
            PYTHONPATH=os.getenv('PYTHONPATH'),
            ATHENA_ENABLE_TRY_RUN="False",
        ),
    )

def AthenaTryRunEnabled():
    return os.getenv('ATHENA_ENABLE_TRY_RUN') not in {
        "0",
        "False",
        "false",
        "OFF"
    }

def GetNeedSkipAndSkipMessage():
    current_stage = GetCurrentCinnStage()
    assert current_stage is not None
    if not IsCinnStageEnableDiff():
        return False, ""
    last_stage = GetPrevCinnStage(current_stage)
    if last_stage is None:
        return False, ""
    exitcode, stderr = GetStageExitCodeAndStdErr(last_stage)
    if exitcode != 0:
        return True, "last stage failed."
    return False, ""

def GetCurrentStageTryRunExitCodeAndStdErr():
    if not AthenaTryRunEnabled():
        return False, ""
    current_stage = GetCurrentCinnStage()
    assert current_stage is not None
    return GetStageExitCodeAndStdErr(current_stage)

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

    def _test_entry(self):
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





need_skip, skip_message = GetNeedSkipAndSkipMessage()
try_run_exit_code, try_run_stderr = GetCurrentStageTryRunExitCodeAndStdErr()
class TestTryRun(unittest.TestCase):
    def test_panic(self):
        if not AthenaTryRunEnabled():
            return
        if try_run_exit_code == 0:
            # All unittest cases passed.
            return
        if try_run_exit_code > 0:
            # program failed but not panic.
            return
        # program panicked.
        kOutputLimit = 65536
        message = try_run_stderr[-kOutputLimit:]
        raise RuntimeError(f"panicked. last {kOutputLimit} characters of stderr: \n{message}")
class PrimitiveOp_329f447a7beacb79da5dba83f1745b02(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1, arg_1_2):
        arg_1_0 = paddle._C_ops.full_int_array(1, paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([-1], paddle.int32, paddle.core.CPUPlace())
        arg_1_2 = paddle._C_ops.full_int_array([-1], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1, arg_1_2]
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_562197110f9f0582cbecacab92dd3253(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_329f447a7beacb79da5dba83f1745b02
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4a89422d71bddb017c2799cc9c0b2ac9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1):
        arg_1_0 = paddle._C_ops.full_int_array(1, paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([26], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8a88fbe0ef63b0a20df82aa3d97a8954(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a89422d71bddb017c2799cc9c0b2ac9
    def get_inputs(self):
        return [
            paddle.to_tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]], dtype='int64').reshape([1, 26]),
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor([26], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_06f6a85b77c9e5c3759ce4bcae1006bb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1, arg_1_2):
        arg_1_0 = paddle._C_ops.full_int_array(1, paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([-1], paddle.int32, paddle.core.CPUPlace())
        arg_1_2 = paddle._C_ops.full_int_array([-1], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1, arg_1_2]
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_70e672e2fddd2ac661c765c9378539cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06f6a85b77c9e5c3759ce4bcae1006bb
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 768], dtype='float16', min=0, max=0.5),
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6992e6277d49682bb6e08c5ac2e7528b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_329f447a7beacb79da5dba83f1745b02
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_68f9efb830d8db3ee590e42ff3e31488(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1, arg_1_2):
        arg_1_0 = paddle._C_ops.full_int_array(1, paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([3], paddle.int32, paddle.core.CPUPlace())
        arg_1_2 = paddle._C_ops.full_int_array([2], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1, arg_1_2]
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_370dabd96232c88d2e35ed3fcdce7144(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_68f9efb830d8db3ee590e42ff3e31488
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.47021484375, 0.061798095703125], [0.298828125, 0.388916015625], [0.420166015625, 0.08502197265625]], dtype='float16').reshape([3, 2]),
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor([3], dtype='int32').reshape([1]),
            paddle.to_tensor([2], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_91eacc80e52a2eaec79efe9671b71d00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06f6a85b77c9e5c3759ce4bcae1006bb
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 384], dtype='float16', min=0, max=0.5),
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ec566d0907dbd94af134ff3e026f8b45(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1, arg_1_2):
        arg_1_0 = paddle._C_ops.full_int_array(1, paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([3], paddle.int32, paddle.core.CPUPlace())
        arg_1_2 = paddle._C_ops.full_int_array([2], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1, arg_1_2]
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_812743694c10f0ef722fcc011cae2295(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec566d0907dbd94af134ff3e026f8b45
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.05006412789225578, 0.3941880464553833], [0.35345566272735596, 0.38984331488609314], [0.32644951343536377, 0.17681334912776947]], dtype='float32').reshape([3, 2]),
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor([3], dtype='int32').reshape([1]),
            paddle.to_tensor([2], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f701ec8a598a929d9057843424d9f7af(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 80, 80, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a343ea37d163ab32ea385b0b162dc227(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f701ec8a598a929d9057843424d9f7af
    def get_inputs(self):
        return [
            paddle.uniform([80, 80, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1, 3, 80, 80, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2c0bb69cd75aa2a0fb18c0f02fb9af6e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 40, 40, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1f05dcc6110708adc21a21dc26f70ab7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c0bb69cd75aa2a0fb18c0f02fb9af6e
    def get_inputs(self):
        return [
            paddle.uniform([40, 40, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1, 3, 40, 40, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9ad1122bd8e713189fdec2e0e12377c2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 20, 20, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_269cc63e2f72a7a395dbb30078221b59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ad1122bd8e713189fdec2e0e12377c2
    def get_inputs(self):
        return [
            paddle.uniform([20, 20, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1, 3, 20, 20, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_aea5b74822a641a85c6867f77afb1b66(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 80, 80, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7e2859887940faa5feabe6e76caa2dc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aea5b74822a641a85c6867f77afb1b66
    def get_inputs(self):
        return [
            paddle.uniform([80, 80, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 3, 80, 80, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_49c335aee2f5e66b99c7439ccada3063(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 40, 40, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a6dc8b4aa79950087ff7a6c1e713afc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_49c335aee2f5e66b99c7439ccada3063
    def get_inputs(self):
        return [
            paddle.uniform([40, 40, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 3, 40, 40, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5f8e349549466c8fbac20c02609943f3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 20, 20, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9131fc61bc15e9f69b8d4d40a9ff428f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f8e349549466c8fbac20c02609943f3
    def get_inputs(self):
        return [
            paddle.uniform([20, 20, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 3, 20, 20, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a488728dab434f44c3229c0368637a49(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1, arg_1_2):
        arg_1_0 = paddle._C_ops.full_int_array(1, paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([-1], paddle.int32, paddle.core.CPUPlace())
        arg_1_2 = paddle._C_ops.full_int_array([-1], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1, arg_1_2]
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f4262f7bc57b7dbb77b64fa129fbc547(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a488728dab434f44c3229c0368637a49
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1c0935a2d2451debf76c93cf99ff36e7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1):
        arg_1_0 = paddle._C_ops.full_int_array(1, paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([26], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 26], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_59afbdfa8ebe14efd4172b6ae253be51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c0935a2d2451debf76c93cf99ff36e7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]], dtype='int64').reshape([1, 26]),
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor([26], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_add2e5c53e21e99fead57290a15c4fa5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1, arg_1_2):
        arg_1_0 = paddle._C_ops.full_int_array(1, paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([-1], paddle.int32, paddle.core.CPUPlace())
        arg_1_2 = paddle._C_ops.full_int_array([-1], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1, arg_1_2]
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 768], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1a4cafdc657d8753778478c81a370645(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_add2e5c53e21e99fead57290a15c4fa5
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 768], dtype='float16', min=0, max=0.5),
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4474fd0c8438f504b7cdb1214a43b41a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1, arg_1_2):
        arg_1_0 = paddle._C_ops.full_int_array(1, paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([-1], paddle.int32, paddle.core.CPUPlace())
        arg_1_2 = paddle._C_ops.full_int_array([-1], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1, arg_1_2]
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cbfa69cd3d67411c05110c5315a4c809(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4474fd0c8438f504b7cdb1214a43b41a
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_73eca443be8aec60e04259d7ec5c23b1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1, arg_1_2):
        arg_1_0 = paddle._C_ops.full_int_array(1, paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([3], paddle.int32, paddle.core.CPUPlace())
        arg_1_2 = paddle._C_ops.full_int_array([2], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1, arg_1_2]
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 2], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5bb172487fce0d035e1c660fc9783235(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73eca443be8aec60e04259d7ec5c23b1
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.47021484375, 0.061798095703125], [0.298828125, 0.388916015625], [0.420166015625, 0.08502197265625]], dtype='float16').reshape([3, 2]),
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor([3], dtype='int32').reshape([1]),
            paddle.to_tensor([2], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_76a98f622dce229aaadb6c47d84809fa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1, arg_1_2):
        arg_1_0 = paddle._C_ops.full_int_array(1, paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([-1], paddle.int32, paddle.core.CPUPlace())
        arg_1_2 = paddle._C_ops.full_int_array([-1], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1, arg_1_2]
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 384], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3b0b407f6f6d9e37d52d8476f891c679(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76a98f622dce229aaadb6c47d84809fa
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 384], dtype='float16', min=0, max=0.5),
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5a29f9ab6770b966ccc7325427b8e85f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1, arg_1_2):
        arg_1_0 = paddle._C_ops.full_int_array(1, paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([3], paddle.int32, paddle.core.CPUPlace())
        arg_1_2 = paddle._C_ops.full_int_array([2], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1, arg_1_2]
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d7728d242f811a7fa08e95eb87a6e9af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a29f9ab6770b966ccc7325427b8e85f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.05006412789225578, 0.3941880464553833], [0.35345566272735596, 0.38984331488609314], [0.32644951343536377, 0.17681334912776947]], dtype='float32').reshape([3, 2]),
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor([3], dtype='int32').reshape([1]),
            paddle.to_tensor([2], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6bdce6b84bfd67002cf04115b98a5369(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 80, 80, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[80, 80, 2], dtype='float16'),
            paddle.static.InputSpec(shape=[5], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_31180049b1595e94f0319f5afc7332ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bdce6b84bfd67002cf04115b98a5369
    def get_inputs(self):
        return [
            paddle.uniform([80, 80, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1, 3, 80, 80, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_cc3a7fa1f6597a9d865009233b5588b5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 40, 40, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[40, 40, 2], dtype='float16'),
            paddle.static.InputSpec(shape=[5], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_559bd8186c2182c9ed52f90a28a3b19b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc3a7fa1f6597a9d865009233b5588b5
    def get_inputs(self):
        return [
            paddle.uniform([40, 40, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1, 3, 40, 40, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1c42319fbe57fddaf3568ae679d76a8a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 20, 20, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20, 20, 2], dtype='float16'),
            paddle.static.InputSpec(shape=[5], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1dcefd634df17f6e347548385c5a9fa3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c42319fbe57fddaf3568ae679d76a8a
    def get_inputs(self):
        return [
            paddle.uniform([20, 20, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1, 3, 20, 20, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4a81d798c54b24f330fc800fee557d31(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 80, 80, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[80, 80, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[5], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_111aa266cc5dd64605acb7846be2d9ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a81d798c54b24f330fc800fee557d31
    def get_inputs(self):
        return [
            paddle.uniform([80, 80, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 3, 80, 80, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d9ff143e88931cc7fb0b952581ff1de6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 40, 40, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[40, 40, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[5], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_922275da1e4ae3a338403d2854a843c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9ff143e88931cc7fb0b952581ff1de6
    def get_inputs(self):
        return [
            paddle.uniform([40, 40, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 3, 40, 40, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_da75383804267717813edb3dc1903e62(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 20, 20, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20, 20, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[5], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_215864605da4de03c1e3d595b915be2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da75383804267717813edb3dc1903e62
    def get_inputs(self):
        return [
            paddle.uniform([20, 20, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 3, 20, 20, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()


if __name__ == '__main__':
    unittest.main()