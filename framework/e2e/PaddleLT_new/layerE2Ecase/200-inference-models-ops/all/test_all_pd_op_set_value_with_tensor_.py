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
class PrimitiveOp_b92d730b13f227b646d280bbd5d1a030(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([2147483647], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4a3922e0fc1e503a81b67eafac914326(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b92d730b13f227b646d280bbd5d1a030
    def get_inputs(self):
        return [
            paddle.uniform([1, 197, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2147483647], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_90914a5af1f97437d66db4b7c67ceb2d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([27], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_47c4fd16aaf71d35f77f6941a554a6c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90914a5af1f97437d66db4b7c67ceb2d
    def get_inputs(self):
        return [
            paddle.uniform([1, 38, 56, 56], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 27, 56, 56], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([27], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_ed32a39cf61fdcf4a0b54c656872c21c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([50], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_506674367d3c6583a351be2ce4538b92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ed32a39cf61fdcf4a0b54c656872c21c
    def get_inputs(self):
        return [
            paddle.uniform([1, 61, 28, 28], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 50, 28, 28], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([50], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_018cf003b85195b228f622831c1fd4a9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([72], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b9987792c3034dd0d4b0189cf23e31cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_018cf003b85195b228f622831c1fd4a9
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 72, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([72], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_8521567e599ba829e76e34a52bca9950(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([84], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d27af385df4f8b05336aa9e1fa8a1ec2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8521567e599ba829e76e34a52bca9950
    def get_inputs(self):
        return [
            paddle.uniform([1, 95, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 84, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([84], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_6e525b9d34c025e3448ddb8221438807(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([95], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_23edb71b6b64694f8c702cf57ad0118b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e525b9d34c025e3448ddb8221438807
    def get_inputs(self):
        return [
            paddle.uniform([1, 106, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 95, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([95], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_977c4c3b43fe821b398f52b708d01b46(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([106], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0c7e398c66f2779eb9d2b9e8eb166d77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_977c4c3b43fe821b398f52b708d01b46
    def get_inputs(self):
        return [
            paddle.uniform([1, 117, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 106, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([106], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_0f440d4ef61093dde4b870c294647e07(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([117], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0bb1b82cfc820645f9d933cccee648fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f440d4ef61093dde4b870c294647e07
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 117, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([117], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_56c8d2e34896c2da976b8428eed4399f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([140], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_16f7ff5189ce51e09fc107217cbc1d3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56c8d2e34896c2da976b8428eed4399f
    def get_inputs(self):
        return [
            paddle.uniform([1, 151, 7, 7], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 140, 7, 7], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([140], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_3415865134adfb72bf6404d4576c6b2a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([151], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_80692fd39e65499f9b8c3f4bb34f4f29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3415865134adfb72bf6404d4576c6b2a
    def get_inputs(self):
        return [
            paddle.uniform([1, 162, 7, 7], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 151, 7, 7], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([151], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_7357045aeace5adfbf1791d615006658(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([162], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d87c22d5f3b9063dc9efa246237be1f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7357045aeace5adfbf1791d615006658
    def get_inputs(self):
        return [
            paddle.uniform([1, 174, 7, 7], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 162, 7, 7], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([162], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_824328b35e5ae2449fbab0a7bd882406(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([174], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1d2b2ba7cae36c1c3480be6d9432ac59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_824328b35e5ae2449fbab0a7bd882406
    def get_inputs(self):
        return [
            paddle.uniform([1, 185, 7, 7], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 174, 7, 7], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([174], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_5041aa251dd1933c19b5d5cc946ef3bb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([27], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3a040cf29a83099e7a2ecf97576336d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5041aa251dd1933c19b5d5cc946ef3bb
    def get_inputs(self):
        return [
            paddle.uniform([1, 38, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 27, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([27], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_61acee52101052d336ab476e1182b8fd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([50], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cee0fd7b626d3f598923535c93b53be8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61acee52101052d336ab476e1182b8fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 61, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 50, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([50], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_b238c36e6290ba676ba177e1490b0807(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([72], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b04a4cf3d4be716c0819fddfbeb05f80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b238c36e6290ba676ba177e1490b0807
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([72], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_29fab27cc15d4f1a56fcfeb41bde9b2c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([84], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0e761dfd531690c432b8a9b0080a22af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29fab27cc15d4f1a56fcfeb41bde9b2c
    def get_inputs(self):
        return [
            paddle.uniform([1, 95, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 84, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([84], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_f339c46037b2ad83797ba492701b0b3f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([95], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bd155a4a9f666c7003b814535b9f41c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f339c46037b2ad83797ba492701b0b3f
    def get_inputs(self):
        return [
            paddle.uniform([1, 106, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 95, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([95], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_301297a3a5bd48ba25c0d05f7ae2c1d2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([106], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fb19e2afcc90bd475d3db1ad56b13205(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_301297a3a5bd48ba25c0d05f7ae2c1d2
    def get_inputs(self):
        return [
            paddle.uniform([1, 117, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 106, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([106], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_5a8544daa995eabfde9a73f7466e4737(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([117], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5f33ad5eac3b70db85470adf1b8186b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a8544daa995eabfde9a73f7466e4737
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 117, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([117], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_24ea5f778990f35dcf676df03eaf7ca6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([140], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9c50233225154f0bd80314b058757221(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24ea5f778990f35dcf676df03eaf7ca6
    def get_inputs(self):
        return [
            paddle.uniform([1, 151, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 140, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([140], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_b205d9781cf4cc3d0bc619ebecc1c73f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([151], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_15347d01f67cf407407933a32133ad3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b205d9781cf4cc3d0bc619ebecc1c73f
    def get_inputs(self):
        return [
            paddle.uniform([1, 162, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 151, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([151], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_85a6d1f49eaf76ac12adb6d129f7f85e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([162], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_52a448c0de2c53c0a9ae659a9118cb71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85a6d1f49eaf76ac12adb6d129f7f85e
    def get_inputs(self):
        return [
            paddle.uniform([1, 174, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 162, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([162], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_3d4bab8913c0fcbe0b60fb06064533be(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([174], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f4b42ab781409eec12fa38776f69878d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d4bab8913c0fcbe0b60fb06064533be
    def get_inputs(self):
        return [
            paddle.uniform([1, 185, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 174, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([174], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_f846d8b20964e65a91d848344c4a58df(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([2147483647], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2ac10a2cc8de9f0542d2e2c776f69765(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f846d8b20964e65a91d848344c4a58df
    def get_inputs(self):
        return [
            paddle.uniform([1, 197, 384], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 196, 384], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2147483647], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_db8dbdb81d9e8459b8660f4ab61e4ba3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_661f626633673275db2442c57816b3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db8dbdb81d9e8459b8660f4ab61e4ba3
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_49d184616c5ecb8e390f48b95eb5a43f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([2], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9ea737a2a7e613f8923b1ad6f212554b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_49d184616c5ecb8e390f48b95eb5a43f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_edea654a537a6e7deed2783f49979f60(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fc3dd6fa1cf9b0278cadc739ee6fd326(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edea654a537a6e7deed2783f49979f60
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 180, 320], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 180, 320], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_35f053dfff39b4ae10941b2aaba25695(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([2], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a56f9bc926f7bac06040176685d2c600(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35f053dfff39b4ae10941b2aaba25695
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 180, 320], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 180, 320], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_a288794ca8535511625890d8c260cbb9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([2147483647], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 197, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 196, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3e91de7c5935b5ffd8794940237d30bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a288794ca8535511625890d8c260cbb9
    def get_inputs(self):
        return [
            paddle.uniform([1, 197, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2147483647], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_333af1ccde4b5f0441e03091b4222e6a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([27], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 38, 56, 56], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 27, 56, 56], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9d9116a53f382662fc88e4938ac54ac5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_333af1ccde4b5f0441e03091b4222e6a
    def get_inputs(self):
        return [
            paddle.uniform([1, 38, 56, 56], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 27, 56, 56], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([27], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_6a33eb9486d87fc1b33be01696c59fd1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([50], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 61, 28, 28], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 50, 28, 28], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b45e9d130c5217984aae72e72f31fd9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a33eb9486d87fc1b33be01696c59fd1
    def get_inputs(self):
        return [
            paddle.uniform([1, 61, 28, 28], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 50, 28, 28], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([50], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_8814ad6b3594421b2973b0431a37ffc2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([72], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 84, 14, 14], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 72, 14, 14], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5d4d758c7eab9e0f31b45a9574083ede(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8814ad6b3594421b2973b0431a37ffc2
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 72, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([72], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_7045262c320f92568f6fc0f6fdb7932a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([84], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 95, 14, 14], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 84, 14, 14], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b3cf8b8d8c24320efcbabc74f9f81147(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7045262c320f92568f6fc0f6fdb7932a
    def get_inputs(self):
        return [
            paddle.uniform([1, 95, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 84, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([84], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_fb92aff7f4b60b7d33918906d7cc65bf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([95], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 106, 14, 14], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 95, 14, 14], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_776b688dc92cbba4e1f7d7add28457c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb92aff7f4b60b7d33918906d7cc65bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 106, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 95, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([95], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_976bd62af4c572b0ff775763cdff82bd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([106], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 117, 14, 14], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 106, 14, 14], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2761362cf47dc1d5fa16e6efe49dc89e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_976bd62af4c572b0ff775763cdff82bd
    def get_inputs(self):
        return [
            paddle.uniform([1, 117, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 106, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([106], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_748cba3aeedf3de84cedc0d6835619d1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([117], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 14, 14], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 117, 14, 14], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5bdc2fa547939859aacb824960ce37c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_748cba3aeedf3de84cedc0d6835619d1
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 117, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([117], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_67bf494674428b1657e23724cae48143(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([140], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 151, 7, 7], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 140, 7, 7], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f1cada9aeba709e4daf016c4fbf13041(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67bf494674428b1657e23724cae48143
    def get_inputs(self):
        return [
            paddle.uniform([1, 151, 7, 7], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 140, 7, 7], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([140], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_99e8ec6e1e86072ac2b1abaaf541a86c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([151], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 162, 7, 7], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 151, 7, 7], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_696bda10b106a38923bff441a28779a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99e8ec6e1e86072ac2b1abaaf541a86c
    def get_inputs(self):
        return [
            paddle.uniform([1, 162, 7, 7], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 151, 7, 7], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([151], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_0d4e6fbf96d94fab6ec8f9db1f238639(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([162], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 174, 7, 7], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 162, 7, 7], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b9e03548035ed735dcfa88d8b6e72ada(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d4e6fbf96d94fab6ec8f9db1f238639
    def get_inputs(self):
        return [
            paddle.uniform([1, 174, 7, 7], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 162, 7, 7], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([162], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_bb4ab3921868a5cd3d78f4df716223af(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([174], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 185, 7, 7], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 174, 7, 7], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5d2ce6a5356c4f86ae378cf2f4ae01d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb4ab3921868a5cd3d78f4df716223af
    def get_inputs(self):
        return [
            paddle.uniform([1, 185, 7, 7], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 174, 7, 7], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([174], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_d42e0627e030ac2f43e38e4b5d4f3b50(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([27], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 38, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 27, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_50ae615d73d15dc5c5531efb005bfd6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d42e0627e030ac2f43e38e4b5d4f3b50
    def get_inputs(self):
        return [
            paddle.uniform([1, 38, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 27, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([27], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_0addc2aac3086886ac03e77673a6241b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([50], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 61, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 50, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3b0d1d66305acde288fb63a366341f7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0addc2aac3086886ac03e77673a6241b
    def get_inputs(self):
        return [
            paddle.uniform([1, 61, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 50, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([50], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_589fe386537ac83f06037d061855479e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([72], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 84, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 72, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_453fb9d2310ae397fc9431a56f5ccde9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_589fe386537ac83f06037d061855479e
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([72], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_0d6ef7179120eab60a904a8c5fb90559(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([84], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 95, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 84, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f2df16ca37445a266b54f64b4aa72dcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d6ef7179120eab60a904a8c5fb90559
    def get_inputs(self):
        return [
            paddle.uniform([1, 95, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 84, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([84], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_6de45ceefb2bc59d5943969ef528adff(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([95], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 106, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 95, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5e244b3c40ba20feb88d46e5c6eb7358(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6de45ceefb2bc59d5943969ef528adff
    def get_inputs(self):
        return [
            paddle.uniform([1, 106, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 95, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([95], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_bcd8c29453e35c8d0755b07f97829224(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([106], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 117, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 106, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6d2cabacb3651b58ab0bd03858a59e54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcd8c29453e35c8d0755b07f97829224
    def get_inputs(self):
        return [
            paddle.uniform([1, 117, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 106, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([106], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_b7684b1b3e75cb695066293ec0ffdf3c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([117], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 117, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ddb2bef8f4f28e3d902fd58a192ddba0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7684b1b3e75cb695066293ec0ffdf3c
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 117, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([117], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_dcde93573413f0154e180b2617ac20c3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([140], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 151, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 140, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8228b1bbae78fbbd3c040362aa9fb11a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dcde93573413f0154e180b2617ac20c3
    def get_inputs(self):
        return [
            paddle.uniform([1, 151, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 140, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([140], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_6be392b975eb44578831838a0eaf413e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([151], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 162, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 151, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d8e6c773fc71759af8667bdad7bfb98f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6be392b975eb44578831838a0eaf413e
    def get_inputs(self):
        return [
            paddle.uniform([1, 162, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 151, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([151], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_8489a97391db99fb939cd36b0d494532(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([162], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 174, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 162, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ce3814ef3b15618209d6383d2e518909(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8489a97391db99fb939cd36b0d494532
    def get_inputs(self):
        return [
            paddle.uniform([1, 174, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 162, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([162], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_8a83f487564c7cdde2187f877c31b563(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([174], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 185, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 174, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_51094c67dde27a92ccc7e49bee147689(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a83f487564c7cdde2187f877c31b563
    def get_inputs(self):
        return [
            paddle.uniform([1, 185, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 174, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([174], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_4365375f0a10ad1ba2f5d20272a01af3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([2147483647], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 197, 384], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 196, 384], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_69d80391e271af3b516e4b04a080a054(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4365375f0a10ad1ba2f5d20272a01af3
    def get_inputs(self):
        return [
            paddle.uniform([1, 197, 384], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 196, 384], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2147483647], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_8446f68066cec4a7fe9ef0a98cb3a7f0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 180, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 180, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a568c020ec2f45dd4507f6352612c925(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8446f68066cec4a7fe9ef0a98cb3a7f0
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_6d25852843c70bc490f5d5ffaac9d363(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([2], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 180, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 180, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2a018a159c0f321c2ed6a066ca3db6ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d25852843c70bc490f5d5ffaac9d363
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_3b6503cc182e12de99174a54f20e5414(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([0], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 180, 320], dtype='float16'),
            paddle.static.InputSpec(shape=[1, 180, 320], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fbc99826d1964673609a67bf78952625(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b6503cc182e12de99174a54f20e5414
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 180, 320], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 180, 320], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_f3a382829cd02ff18a8ecad8095bd3c0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        arg_2 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([2], paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 180, 320], dtype='float16'),
            paddle.static.InputSpec(shape=[1, 180, 320], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_da5b4b9bf02a8ff34e8be361b8cd6b97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3a382829cd02ff18a8ecad8095bd3c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 180, 320], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 180, 320], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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