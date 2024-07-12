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
        return True, f"last stage failed. stderr: {stderr}"
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
class PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e482e3da6da05d66134fe676413d12cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e482e3da6da05d66134fe676413d12cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9065fbac3eff0a712f72fa2cadbb0fe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9065fbac3eff0a712f72fa2cadbb0fe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a8edb17a434775d016c6eca19027fe58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a8edb17a434775d016c6eca19027fe58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a1c3488cd25e2c0ae7e38c01be6445a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a1c3488cd25e2c0ae7e38c01be6445a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_974facc602199b7bc1bdae448f5e5dd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_974facc602199b7bc1bdae448f5e5dd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_375d11aa0d5e540fed5bbdd13920b9f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_375d11aa0d5e540fed5bbdd13920b9f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e482e3da6da05d66134fe676413d12cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e482e3da6da05d66134fe676413d12cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2e8adaa438466975c3a3e119004b3b2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2e8adaa438466975c3a3e119004b3b2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4f803e2c8a33161327072e3a6bdc33d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4f803e2c8a33161327072e3a6bdc33d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_388d3ab985a3b2929bc1d537c860a040(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e
    def get_inputs(self):
        return [
            paddle.uniform([1778, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1778, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_388d3ab985a3b2929bc1d537c860a040(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e
    def get_inputs(self):
        return [
            paddle.uniform([1778, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1778, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_388d3ab985a3b2929bc1d537c860a040(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e
    def get_inputs(self):
        return [
            paddle.uniform([1778, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1778, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_388d3ab985a3b2929bc1d537c860a040(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e
    def get_inputs(self):
        return [
            paddle.uniform([1778, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1778, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0cd2878045340812ecf021328275f602(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0cd2878045340812ecf021328275f602(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_07c294ecc40b94e1b1e88bc47a09ed9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_07c294ecc40b94e1b1e88bc47a09ed9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_11bac9787f4a54eb51af7268911a963a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_11bac9787f4a54eb51af7268911a963a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_47df124c02e056bf73339c44d3291633(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_47df124c02e056bf73339c44d3291633(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_444b47dac04850fe98311719eeaecb7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.10817418247461319], [0.21830850839614868], [0.1470475047826767], [0.2576325237751007], [0.06852712482213974], [0.39306512475013733], [0.006903389468789101], [0.1592971235513687], [0.36045563220977783]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.3560943007469177], [0.2843972146511078], [0.19924697279930115], [0.39769965410232544], [0.1095288023352623], [0.4948180615901947], [0.43056219816207886], [0.004259239882230759], [0.4542475640773773]], dtype='float32').reshape([9, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_85f9650b2590638e547573b337f54631(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.013448690064251423], [0.39393821358680725], [0.41981005668640137], [0.2410021275281906], [0.28984448313713074], [0.2892489433288574], [0.022538647055625916], [0.48357874155044556], [0.11527103185653687]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.09306798875331879], [0.1428917944431305], [0.18482916057109833], [0.4609587788581848], [0.11816191673278809], [0.3480241000652313], [0.3102295994758606], [0.42789679765701294], [0.15572763979434967]], dtype='float32').reshape([9, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_349178674db3f1823755a7bc40a6e0be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.16345298290252686], [0.03933706879615784], [0.26740995049476624], [0.15774424374103546], [0.22924834489822388], [0.3102510869503021], [0.4106326401233673], [0.31178075075149536], [0.4077361524105072]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.3870587646961212], [0.0602361299097538], [0.3039933145046234], [0.4867640435695648], [0.405436635017395], [0.246979221701622], [0.08601068705320358], [0.37027329206466675], [0.10850615799427032]], dtype='float32').reshape([9, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_442cbd17a6f2345925002707370b675e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.22238902747631073], [0.048635415732860565], [0.2709272503852844], [0.37756359577178955], [0.17188876867294312], [0.06343194842338562], [0.4906500577926636], [0.49394193291664124], [0.19016608595848083]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.3498869836330414], [0.26301130652427673], [0.3965403735637665], [0.2822781205177307], [0.03485541045665741], [0.011633493937551975], [0.3463441729545593], [0.45662522315979004], [0.43974825739860535]], dtype='float32').reshape([9, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f90d39b0908cb9fbfe71e4f3364b7985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e
    def get_inputs(self):
        return [
            paddle.uniform([5640, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5640, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f90d39b0908cb9fbfe71e4f3364b7985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e
    def get_inputs(self):
        return [
            paddle.uniform([5640, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5640, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f90d39b0908cb9fbfe71e4f3364b7985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e
    def get_inputs(self):
        return [
            paddle.uniform([5640, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5640, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f90d39b0908cb9fbfe71e4f3364b7985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e
    def get_inputs(self):
        return [
            paddle.uniform([5640, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5640, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_880c9d6dc82e08e6e12604b9a4d9df14(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.minimum(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ceab702b515fde6610456770d44229c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_880c9d6dc82e08e6e12604b9a4d9df14
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4102131128311157, 0.35089540481567383, 0.2971991300582886, 0.29094526171684265, 0.40372434258461, 0.39073458313941956], dtype='float32').reshape([6]),
            paddle.to_tensor([0.3418285846710205, 0.42906653881073, 0.04830992594361305, 0.33282768726348877, 0.28770944476127625, 0.12256492674350739], dtype='float32').reshape([6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d4275546764a894aef2692b8df63dd4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_880c9d6dc82e08e6e12604b9a4d9df14
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1450636386871338, 0.25107648968696594, 0.23796823620796204, 0.386831134557724, 0.3538130819797516, 0.40666136145591736], dtype='float32').reshape([6]),
            paddle.to_tensor([0.4553065598011017, 0.16179899871349335, 0.38409265875816345, 0.37823209166526794, 0.21884433925151825, 0.2555798888206482], dtype='float32').reshape([6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c3301e33642ba9f66c5fcd104765179d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_880c9d6dc82e08e6e12604b9a4d9df14
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4102131128311157, 0.047395769506692886, 0.13399538397789001, 0.29094526171684265, 0.009467806667089462, 0.18842779099941254], dtype='float32').reshape([6]),
            paddle.to_tensor([0.3343552052974701, 0.20509734749794006, 0.16082343459129333, 0.4748803675174713, 0.07652277499437332, 0.20391766726970673], dtype='float32').reshape([6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5c406e1e73481e753773dfd8d46b1670(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_880c9d6dc82e08e6e12604b9a4d9df14
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1450636386871338, 0.25107648968696594, 0.23796823620796204, 0.386831134557724, 0.3538130819797516, 0.40666136145591736], dtype='float32').reshape([6]),
            paddle.to_tensor([0.4793972969055176, 0.15416307747364044, 0.08819928020238876, 0.2687219977378845, 0.017487674951553345, 0.48301398754119873], dtype='float32').reshape([6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6f4ff1ed251b0a4435d9179be0857e84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e
    def get_inputs(self):
        return [
            paddle.uniform([1797, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1797, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6f4ff1ed251b0a4435d9179be0857e84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e
    def get_inputs(self):
        return [
            paddle.uniform([1797, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1797, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6f4ff1ed251b0a4435d9179be0857e84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e
    def get_inputs(self):
        return [
            paddle.uniform([1797, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1797, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6f4ff1ed251b0a4435d9179be0857e84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e
    def get_inputs(self):
        return [
            paddle.uniform([1797, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1797, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a1c3488cd25e2c0ae7e38c01be6445a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a1c3488cd25e2c0ae7e38c01be6445a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_07c294ecc40b94e1b1e88bc47a09ed9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_07c294ecc40b94e1b1e88bc47a09ed9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_41db21b082fdab813ba34e6df806160c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e
    def get_inputs(self):
        return [
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_41db21b082fdab813ba34e6df806160c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e
    def get_inputs(self):
        return [
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_41db21b082fdab813ba34e6df806160c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e
    def get_inputs(self):
        return [
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_41db21b082fdab813ba34e6df806160c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e
    def get_inputs(self):
        return [
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_375d11aa0d5e540fed5bbdd13920b9f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_375d11aa0d5e540fed5bbdd13920b9f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9065fbac3eff0a712f72fa2cadbb0fe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9065fbac3eff0a712f72fa2cadbb0fe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_29fd5f7f137684994409558df26e76a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.26837071776390076]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.16136038303375244]], dtype='float32').reshape([1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5594ced588a4bb616bb0838cd69d3132(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.43908387422561646]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.4695817530155182]], dtype='float32').reshape([1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8af6b6a0970cd7eeb9e1eb21125c5743(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.08844736218452454]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.4814324676990509]], dtype='float32').reshape([1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b76b533586bf2562c6c736ba84198a36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.235838383436203]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.22826553881168365]], dtype='float32').reshape([1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e060626f77a4c1ea32463787011e314b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.35055011510849], [0.09527037292718887], [0.40782997012138367], [0.12946005165576935], [0.37342819571495056], [0.3238355815410614]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.355660617351532], [0.35937029123306274], [0.32676854729652405], [0.3721979558467865], [0.4757102429866791], [0.03219883143901825]], dtype='float32').reshape([6, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_112117256dc842aeab732cede6676b31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.405924528837204], [0.31001538038253784], [0.2668556869029999], [0.47325971722602844], [0.4950177073478699], [0.3854176700115204]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.43543845415115356], [0.21171081066131592], [0.1269102543592453], [0.3708653151988983], [0.14785419404506683], [0.4037357568740845]], dtype='float32').reshape([6, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_590173a947d667b2afc438c1312e1f4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3338686227798462], [0.0581473633646965], [0.27728891372680664], [0.40559735894203186], [0.2994241714477539], [0.46134132146835327]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.06195531785488129], [0.07760456204414368], [0.3587840497493744], [0.39517515897750854], [0.3328937888145447], [0.35186487436294556]], dtype='float32').reshape([6, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0fbae71932afc41ccc23fc7e6704706b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.39591267704963684], [0.11558108776807785], [0.3512172996997833], [0.486686110496521], [0.46994784474372864], [0.3137425482273102]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.11733926832675934], [0.25403302907943726], [0.09014913439750671], [0.011464092880487442], [0.316177636384964], [0.44575998187065125]], dtype='float32').reshape([6, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0cd2878045340812ecf021328275f602(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0cd2878045340812ecf021328275f602(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a8edb17a434775d016c6eca19027fe58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a8edb17a434775d016c6eca19027fe58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_65d3b57d8f95693e30ee6a882f8570c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e
    def get_inputs(self):
        return [
            paddle.uniform([2078, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2078, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_65d3b57d8f95693e30ee6a882f8570c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e
    def get_inputs(self):
        return [
            paddle.uniform([2078, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2078, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_65d3b57d8f95693e30ee6a882f8570c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e
    def get_inputs(self):
        return [
            paddle.uniform([2078, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2078, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_65d3b57d8f95693e30ee6a882f8570c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e
    def get_inputs(self):
        return [
            paddle.uniform([2078, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2078, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bf9adabe537b7e09d5e03c3fc444c50e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bf9adabe537b7e09d5e03c3fc444c50e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_815fffef2310b317c5c093561d79d2a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e
    def get_inputs(self):
        return [
            paddle.uniform([4653, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4653, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_815fffef2310b317c5c093561d79d2a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e
    def get_inputs(self):
        return [
            paddle.uniform([4653, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4653, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_815fffef2310b317c5c093561d79d2a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e
    def get_inputs(self):
        return [
            paddle.uniform([4653, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4653, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_815fffef2310b317c5c093561d79d2a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e
    def get_inputs(self):
        return [
            paddle.uniform([4653, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4653, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b97e5287a24b14ade04efda754e1a144(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e
    def get_inputs(self):
        return [
            paddle.uniform([1022, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1022, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b97e5287a24b14ade04efda754e1a144(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e
    def get_inputs(self):
        return [
            paddle.uniform([1022, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1022, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b97e5287a24b14ade04efda754e1a144(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e
    def get_inputs(self):
        return [
            paddle.uniform([1022, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1022, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b97e5287a24b14ade04efda754e1a144(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e
    def get_inputs(self):
        return [
            paddle.uniform([1022, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1022, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_974facc602199b7bc1bdae448f5e5dd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_974facc602199b7bc1bdae448f5e5dd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_11bac9787f4a54eb51af7268911a963a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_11bac9787f4a54eb51af7268911a963a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ab3f4976f12df9c244f1a1ec06b6f2c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.19564540684223175], [0.034317974001169205], [0.29465341567993164], [0.24096974730491638], [0.33738017082214355]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.37746039032936096], [0.16417308151721954], [0.38406670093536377], [0.27725130319595337], [0.4067143201828003]], dtype='float32').reshape([5, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_897547512131572bad5570abe9601512(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.12423448264598846], [0.3562788665294647], [0.11915521323680878], [0.15779650211334229], [0.047968797385692596]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.034618474543094635], [0.3611237704753876], [0.4964972734451294], [0.23624840378761292], [0.08820351958274841]], dtype='float32').reshape([5, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c3f69db5cd72870ce93f6aa3e549aae2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.20026913285255432], [0.18348328769207], [0.18956781923770905], [0.2602648437023163], [0.2216607630252838]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.07452657073736191], [0.41332828998565674], [0.12519223988056183], [0.003253175877034664], [0.48564788699150085]], dtype='float32').reshape([5, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f5b07c0bfefb0d79ec78025c6e94f1db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.22313657402992249], [0.19915108382701874], [0.2620335519313812], [0.48791664838790894], [0.14740948379039764]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.00723513076081872], [0.2731638550758362], [0.16522662341594696], [0.19373512268066406], [0.16293737292289734]], dtype='float32').reshape([5, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4f803e2c8a33161327072e3a6bdc33d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4f803e2c8a33161327072e3a6bdc33d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_47df124c02e056bf73339c44d3291633(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_47df124c02e056bf73339c44d3291633(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_38235bc0a705c72d79d81232af7df08c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_38235bc0a705c72d79d81232af7df08c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_428dce9e41bd825eaae0b8ce7a292986(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e
    def get_inputs(self):
        return [
            paddle.uniform([2305, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2305, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_428dce9e41bd825eaae0b8ce7a292986(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e
    def get_inputs(self):
        return [
            paddle.uniform([2305, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2305, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_428dce9e41bd825eaae0b8ce7a292986(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e
    def get_inputs(self):
        return [
            paddle.uniform([2305, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2305, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_428dce9e41bd825eaae0b8ce7a292986(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e
    def get_inputs(self):
        return [
            paddle.uniform([2305, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2305, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9c6e4ce769fe857bf5dd346f62159253(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e
    def get_inputs(self):
        return [
            paddle.uniform([3053, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3053, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9c6e4ce769fe857bf5dd346f62159253(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e
    def get_inputs(self):
        return [
            paddle.uniform([3053, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3053, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9c6e4ce769fe857bf5dd346f62159253(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e
    def get_inputs(self):
        return [
            paddle.uniform([3053, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3053, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9c6e4ce769fe857bf5dd346f62159253(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e
    def get_inputs(self):
        return [
            paddle.uniform([3053, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3053, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_42254278f18df11a13eec1150cd6cd6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e
    def get_inputs(self):
        return [
            paddle.uniform([3712, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3712, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_42254278f18df11a13eec1150cd6cd6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e
    def get_inputs(self):
        return [
            paddle.uniform([3712, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3712, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_42254278f18df11a13eec1150cd6cd6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e
    def get_inputs(self):
        return [
            paddle.uniform([3712, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3712, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_42254278f18df11a13eec1150cd6cd6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e
    def get_inputs(self):
        return [
            paddle.uniform([3712, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3712, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bf9adabe537b7e09d5e03c3fc444c50e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bf9adabe537b7e09d5e03c3fc444c50e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_81c2c9adebe83d4365a35f20d3f87e52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.11194626986980438], [0.3971475064754486], [0.26984596252441406], [0.32563695311546326]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.03196848928928375], [0.24156440794467926], [0.0597226619720459], [0.4864690601825714]], dtype='float32').reshape([4, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e48ae772e7cbfd5eb92f5d38a57450cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06388451904058456], [0.2715681791305542], [0.12175875902175903], [0.30955228209495544]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.17564170062541962], [0.14793279767036438], [0.27478042244911194], [0.26954853534698486]], dtype='float32').reshape([4, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9f51a9d5e398490e07134104fe6fe52b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2075873762369156], [0.3073572814464569], [0.30402541160583496], [0.26391908526420593]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.07909782230854034], [0.09549641609191895], [0.05401938408613205], [0.30721813440322876]], dtype='float32').reshape([4, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ea2cbb347dd0ab0921624e15c1bf3d14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2821217179298401], [0.4224745035171509], [0.2759237587451935], [0.07581711560487747]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.45928049087524414], [0.4047701358795166], [0.48921844363212585], [0.4449281692504883]], dtype='float32').reshape([4, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_863055afa3185f890ba33c427ee81a9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e
    def get_inputs(self):
        return [
            paddle.uniform([2102, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2102, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_863055afa3185f890ba33c427ee81a9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e
    def get_inputs(self):
        return [
            paddle.uniform([2102, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2102, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_863055afa3185f890ba33c427ee81a9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e
    def get_inputs(self):
        return [
            paddle.uniform([2102, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2102, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_863055afa3185f890ba33c427ee81a9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e
    def get_inputs(self):
        return [
            paddle.uniform([2102, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2102, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_38235bc0a705c72d79d81232af7df08c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_38235bc0a705c72d79d81232af7df08c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2e8adaa438466975c3a3e119004b3b2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2e8adaa438466975c3a3e119004b3b2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1b89b2985e6bbcfb66f820c8b93b2ef1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1b89b2985e6bbcfb66f820c8b93b2ef1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5b85e1ada7b7d9f2c4e1173f50798a2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e
    def get_inputs(self):
        return [
            paddle.uniform([4228, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4228, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5b85e1ada7b7d9f2c4e1173f50798a2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e
    def get_inputs(self):
        return [
            paddle.uniform([4228, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4228, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5b85e1ada7b7d9f2c4e1173f50798a2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e
    def get_inputs(self):
        return [
            paddle.uniform([4228, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4228, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5b85e1ada7b7d9f2c4e1173f50798a2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986339ebe19a912dee0d5ff9f7dfa95e
    def get_inputs(self):
        return [
            paddle.uniform([4228, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4228, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1b89b2985e6bbcfb66f820c8b93b2ef1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1b89b2985e6bbcfb66f820c8b93b2ef1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e482e3da6da05d66134fe676413d12cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e482e3da6da05d66134fe676413d12cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9065fbac3eff0a712f72fa2cadbb0fe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9065fbac3eff0a712f72fa2cadbb0fe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a8edb17a434775d016c6eca19027fe58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a8edb17a434775d016c6eca19027fe58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a1c3488cd25e2c0ae7e38c01be6445a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a1c3488cd25e2c0ae7e38c01be6445a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_974facc602199b7bc1bdae448f5e5dd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_974facc602199b7bc1bdae448f5e5dd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_375d11aa0d5e540fed5bbdd13920b9f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_375d11aa0d5e540fed5bbdd13920b9f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e482e3da6da05d66134fe676413d12cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e482e3da6da05d66134fe676413d12cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2e8adaa438466975c3a3e119004b3b2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2e8adaa438466975c3a3e119004b3b2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4f803e2c8a33161327072e3a6bdc33d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4f803e2c8a33161327072e3a6bdc33d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_95ea32cba3b6bac59a10854970ab22fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.uniform([1778, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1778, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_95ea32cba3b6bac59a10854970ab22fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.uniform([1778, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1778, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_95ea32cba3b6bac59a10854970ab22fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.uniform([1778, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1778, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_95ea32cba3b6bac59a10854970ab22fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.uniform([1778, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1778, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0cd2878045340812ecf021328275f602(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0cd2878045340812ecf021328275f602(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_07c294ecc40b94e1b1e88bc47a09ed9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_07c294ecc40b94e1b1e88bc47a09ed9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_11bac9787f4a54eb51af7268911a963a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_11bac9787f4a54eb51af7268911a963a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_47df124c02e056bf73339c44d3291633(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_47df124c02e056bf73339c44d3291633(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_444b47dac04850fe98311719eeaecb7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.10817418247461319], [0.21830850839614868], [0.1470475047826767], [0.2576325237751007], [0.06852712482213974], [0.39306512475013733], [0.006903389468789101], [0.1592971235513687], [0.36045563220977783]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.3560943007469177], [0.2843972146511078], [0.19924697279930115], [0.39769965410232544], [0.1095288023352623], [0.4948180615901947], [0.43056219816207886], [0.004259239882230759], [0.4542475640773773]], dtype='float32').reshape([9, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_85f9650b2590638e547573b337f54631(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.013448690064251423], [0.39393821358680725], [0.41981005668640137], [0.2410021275281906], [0.28984448313713074], [0.2892489433288574], [0.022538647055625916], [0.48357874155044556], [0.11527103185653687]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.09306798875331879], [0.1428917944431305], [0.18482916057109833], [0.4609587788581848], [0.11816191673278809], [0.3480241000652313], [0.3102295994758606], [0.42789679765701294], [0.15572763979434967]], dtype='float32').reshape([9, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_349178674db3f1823755a7bc40a6e0be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.16345298290252686], [0.03933706879615784], [0.26740995049476624], [0.15774424374103546], [0.22924834489822388], [0.3102510869503021], [0.4106326401233673], [0.31178075075149536], [0.4077361524105072]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.3870587646961212], [0.0602361299097538], [0.3039933145046234], [0.4867640435695648], [0.405436635017395], [0.246979221701622], [0.08601068705320358], [0.37027329206466675], [0.10850615799427032]], dtype='float32').reshape([9, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_442cbd17a6f2345925002707370b675e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.22238902747631073], [0.048635415732860565], [0.2709272503852844], [0.37756359577178955], [0.17188876867294312], [0.06343194842338562], [0.4906500577926636], [0.49394193291664124], [0.19016608595848083]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.3498869836330414], [0.26301130652427673], [0.3965403735637665], [0.2822781205177307], [0.03485541045665741], [0.011633493937551975], [0.3463441729545593], [0.45662522315979004], [0.43974825739860535]], dtype='float32').reshape([9, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_efdd7ca934d4e58a2d5579b0658230ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.uniform([5640, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5640, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_efdd7ca934d4e58a2d5579b0658230ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.uniform([5640, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5640, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_efdd7ca934d4e58a2d5579b0658230ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.uniform([5640, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5640, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_efdd7ca934d4e58a2d5579b0658230ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.uniform([5640, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5640, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ceab702b515fde6610456770d44229c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_880c9d6dc82e08e6e12604b9a4d9df14
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4102131128311157, 0.35089540481567383, 0.2971991300582886, 0.29094526171684265, 0.40372434258461, 0.39073458313941956], dtype='float32').reshape([6]),
            paddle.to_tensor([0.3418285846710205, 0.42906653881073, 0.04830992594361305, 0.33282768726348877, 0.28770944476127625, 0.12256492674350739], dtype='float32').reshape([6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d4275546764a894aef2692b8df63dd4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_880c9d6dc82e08e6e12604b9a4d9df14
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1450636386871338, 0.25107648968696594, 0.23796823620796204, 0.386831134557724, 0.3538130819797516, 0.40666136145591736], dtype='float32').reshape([6]),
            paddle.to_tensor([0.4553065598011017, 0.16179899871349335, 0.38409265875816345, 0.37823209166526794, 0.21884433925151825, 0.2555798888206482], dtype='float32').reshape([6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c3301e33642ba9f66c5fcd104765179d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_880c9d6dc82e08e6e12604b9a4d9df14
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4102131128311157, 0.047395769506692886, 0.13399538397789001, 0.29094526171684265, 0.009467806667089462, 0.18842779099941254], dtype='float32').reshape([6]),
            paddle.to_tensor([0.3343552052974701, 0.20509734749794006, 0.16082343459129333, 0.4748803675174713, 0.07652277499437332, 0.20391766726970673], dtype='float32').reshape([6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5c406e1e73481e753773dfd8d46b1670(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_880c9d6dc82e08e6e12604b9a4d9df14
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1450636386871338, 0.25107648968696594, 0.23796823620796204, 0.386831134557724, 0.3538130819797516, 0.40666136145591736], dtype='float32').reshape([6]),
            paddle.to_tensor([0.4793972969055176, 0.15416307747364044, 0.08819928020238876, 0.2687219977378845, 0.017487674951553345, 0.48301398754119873], dtype='float32').reshape([6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bb3d95ccdb0ee0453466006d3a73aff8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.uniform([1797, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1797, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bb3d95ccdb0ee0453466006d3a73aff8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.uniform([1797, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1797, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bb3d95ccdb0ee0453466006d3a73aff8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.uniform([1797, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1797, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bb3d95ccdb0ee0453466006d3a73aff8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.uniform([1797, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1797, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a1c3488cd25e2c0ae7e38c01be6445a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a1c3488cd25e2c0ae7e38c01be6445a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_07c294ecc40b94e1b1e88bc47a09ed9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_07c294ecc40b94e1b1e88bc47a09ed9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9fdd9a8d0b57b0183cc94eac3b83f47b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9fdd9a8d0b57b0183cc94eac3b83f47b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9fdd9a8d0b57b0183cc94eac3b83f47b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9fdd9a8d0b57b0183cc94eac3b83f47b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1501, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_375d11aa0d5e540fed5bbdd13920b9f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_375d11aa0d5e540fed5bbdd13920b9f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9065fbac3eff0a712f72fa2cadbb0fe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9065fbac3eff0a712f72fa2cadbb0fe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_29fd5f7f137684994409558df26e76a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.26837071776390076]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.16136038303375244]], dtype='float32').reshape([1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5594ced588a4bb616bb0838cd69d3132(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.43908387422561646]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.4695817530155182]], dtype='float32').reshape([1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8af6b6a0970cd7eeb9e1eb21125c5743(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.08844736218452454]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.4814324676990509]], dtype='float32').reshape([1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b76b533586bf2562c6c736ba84198a36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.235838383436203]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.22826553881168365]], dtype='float32').reshape([1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e060626f77a4c1ea32463787011e314b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.35055011510849], [0.09527037292718887], [0.40782997012138367], [0.12946005165576935], [0.37342819571495056], [0.3238355815410614]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.355660617351532], [0.35937029123306274], [0.32676854729652405], [0.3721979558467865], [0.4757102429866791], [0.03219883143901825]], dtype='float32').reshape([6, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_112117256dc842aeab732cede6676b31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.405924528837204], [0.31001538038253784], [0.2668556869029999], [0.47325971722602844], [0.4950177073478699], [0.3854176700115204]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.43543845415115356], [0.21171081066131592], [0.1269102543592453], [0.3708653151988983], [0.14785419404506683], [0.4037357568740845]], dtype='float32').reshape([6, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_590173a947d667b2afc438c1312e1f4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3338686227798462], [0.0581473633646965], [0.27728891372680664], [0.40559735894203186], [0.2994241714477539], [0.46134132146835327]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.06195531785488129], [0.07760456204414368], [0.3587840497493744], [0.39517515897750854], [0.3328937888145447], [0.35186487436294556]], dtype='float32').reshape([6, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0fbae71932afc41ccc23fc7e6704706b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.39591267704963684], [0.11558108776807785], [0.3512172996997833], [0.486686110496521], [0.46994784474372864], [0.3137425482273102]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.11733926832675934], [0.25403302907943726], [0.09014913439750671], [0.011464092880487442], [0.316177636384964], [0.44575998187065125]], dtype='float32').reshape([6, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0cd2878045340812ecf021328275f602(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0cd2878045340812ecf021328275f602(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a8edb17a434775d016c6eca19027fe58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a8edb17a434775d016c6eca19027fe58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b382d21b850e94858ddce5f1e4ec439e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.uniform([2078, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2078, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b382d21b850e94858ddce5f1e4ec439e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.uniform([2078, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2078, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b382d21b850e94858ddce5f1e4ec439e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.uniform([2078, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2078, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b382d21b850e94858ddce5f1e4ec439e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.uniform([2078, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2078, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bf9adabe537b7e09d5e03c3fc444c50e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bf9adabe537b7e09d5e03c3fc444c50e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cd38676ad6e05384e2e925759e2537e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.uniform([4653, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4653, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cd38676ad6e05384e2e925759e2537e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.uniform([4653, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4653, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cd38676ad6e05384e2e925759e2537e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.uniform([4653, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4653, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cd38676ad6e05384e2e925759e2537e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.uniform([4653, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4653, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9e92aa7aa058dcf9d738857cdd693b87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.uniform([1022, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1022, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9e92aa7aa058dcf9d738857cdd693b87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.uniform([1022, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1022, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9e92aa7aa058dcf9d738857cdd693b87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.uniform([1022, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1022, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9e92aa7aa058dcf9d738857cdd693b87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.uniform([1022, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1022, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_974facc602199b7bc1bdae448f5e5dd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_974facc602199b7bc1bdae448f5e5dd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_11bac9787f4a54eb51af7268911a963a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_11bac9787f4a54eb51af7268911a963a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ab3f4976f12df9c244f1a1ec06b6f2c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.19564540684223175], [0.034317974001169205], [0.29465341567993164], [0.24096974730491638], [0.33738017082214355]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.37746039032936096], [0.16417308151721954], [0.38406670093536377], [0.27725130319595337], [0.4067143201828003]], dtype='float32').reshape([5, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_897547512131572bad5570abe9601512(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.12423448264598846], [0.3562788665294647], [0.11915521323680878], [0.15779650211334229], [0.047968797385692596]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.034618474543094635], [0.3611237704753876], [0.4964972734451294], [0.23624840378761292], [0.08820351958274841]], dtype='float32').reshape([5, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c3f69db5cd72870ce93f6aa3e549aae2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.20026913285255432], [0.18348328769207], [0.18956781923770905], [0.2602648437023163], [0.2216607630252838]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.07452657073736191], [0.41332828998565674], [0.12519223988056183], [0.003253175877034664], [0.48564788699150085]], dtype='float32').reshape([5, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f5b07c0bfefb0d79ec78025c6e94f1db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.22313657402992249], [0.19915108382701874], [0.2620335519313812], [0.48791664838790894], [0.14740948379039764]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.00723513076081872], [0.2731638550758362], [0.16522662341594696], [0.19373512268066406], [0.16293737292289734]], dtype='float32').reshape([5, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4f803e2c8a33161327072e3a6bdc33d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4f803e2c8a33161327072e3a6bdc33d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_47df124c02e056bf73339c44d3291633(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_47df124c02e056bf73339c44d3291633(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_38235bc0a705c72d79d81232af7df08c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_38235bc0a705c72d79d81232af7df08c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e42aa4515621106c2a87fbddc5d16900(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.uniform([2305, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2305, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e42aa4515621106c2a87fbddc5d16900(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.uniform([2305, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2305, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e42aa4515621106c2a87fbddc5d16900(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.uniform([2305, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2305, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e42aa4515621106c2a87fbddc5d16900(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.uniform([2305, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2305, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3c41eaead04479d8bcffcd76f383391c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.uniform([3053, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3053, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3c41eaead04479d8bcffcd76f383391c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.uniform([3053, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3053, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3c41eaead04479d8bcffcd76f383391c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.uniform([3053, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3053, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3c41eaead04479d8bcffcd76f383391c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.uniform([3053, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3053, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b5ad51f9ef0cb0234873f5073c847179(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.uniform([3712, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3712, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b5ad51f9ef0cb0234873f5073c847179(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.uniform([3712, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3712, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b5ad51f9ef0cb0234873f5073c847179(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.uniform([3712, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3712, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b5ad51f9ef0cb0234873f5073c847179(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.uniform([3712, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3712, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bf9adabe537b7e09d5e03c3fc444c50e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bf9adabe537b7e09d5e03c3fc444c50e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_81c2c9adebe83d4365a35f20d3f87e52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.11194626986980438], [0.3971475064754486], [0.26984596252441406], [0.32563695311546326]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.03196848928928375], [0.24156440794467926], [0.0597226619720459], [0.4864690601825714]], dtype='float32').reshape([4, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e48ae772e7cbfd5eb92f5d38a57450cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06388451904058456], [0.2715681791305542], [0.12175875902175903], [0.30955228209495544]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.17564170062541962], [0.14793279767036438], [0.27478042244911194], [0.26954853534698486]], dtype='float32').reshape([4, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9f51a9d5e398490e07134104fe6fe52b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2075873762369156], [0.3073572814464569], [0.30402541160583496], [0.26391908526420593]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.07909782230854034], [0.09549641609191895], [0.05401938408613205], [0.30721813440322876]], dtype='float32').reshape([4, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ea2cbb347dd0ab0921624e15c1bf3d14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2821217179298401], [0.4224745035171509], [0.2759237587451935], [0.07581711560487747]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.45928049087524414], [0.4047701358795166], [0.48921844363212585], [0.4449281692504883]], dtype='float32').reshape([4, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9e67f157a7634797620610a635eb8d14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.uniform([2102, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2102, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9e67f157a7634797620610a635eb8d14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.uniform([2102, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2102, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9e67f157a7634797620610a635eb8d14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.uniform([2102, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2102, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9e67f157a7634797620610a635eb8d14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.uniform([2102, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2102, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_38235bc0a705c72d79d81232af7df08c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_38235bc0a705c72d79d81232af7df08c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2e8adaa438466975c3a3e119004b3b2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2e8adaa438466975c3a3e119004b3b2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1b89b2985e6bbcfb66f820c8b93b2ef1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1b89b2985e6bbcfb66f820c8b93b2ef1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_75e7bf46aa5fd9c397284d916b98f1bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.uniform([4228, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4228, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_75e7bf46aa5fd9c397284d916b98f1bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.uniform([4228, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4228, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_75e7bf46aa5fd9c397284d916b98f1bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.uniform([4228, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4228, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_75e7bf46aa5fd9c397284d916b98f1bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da1cf5c88aa23daa12f4a9abd711b393
    def get_inputs(self):
        return [
            paddle.uniform([4228, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4228, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1b89b2985e6bbcfb66f820c8b93b2ef1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1b89b2985e6bbcfb66f820c8b93b2ef1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47fe88174ca695e95d3b80825d4b6f42
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()



if __name__ == '__main__':
    unittest.main()