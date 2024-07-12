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
class PrimitiveOp_7c3b0bfbd1c337504fb396997e2c4582(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.floor(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dc7004113544dd97a45a081987b0bb8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c3b0bfbd1c337504fb396997e2c4582
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.9979315996170044]]], [[[1.4513835906982422]]], [[[1.4703072309494019]]], [[[1.514841914176941]]], [[[1.5136427879333496]]], [[[1.5811207294464111]]], [[[1.4758710861206055]]], [[[1.6939473152160645]]], [[[1.8386613130569458]]], [[[1.5455338954925537]]], [[[1.7069263458251953]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_13f75fc3ab118dfff0dbcffbdd74f2a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c3b0bfbd1c337504fb396997e2c4582
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_13f75fc3ab118dfff0dbcffbdd74f2a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c3b0bfbd1c337504fb396997e2c4582
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_a2b7f00520567e6075eb49f16358314c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.floor(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b6c87ee83fe99b8c85301f9673aa385b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2b7f00520567e6075eb49f16358314c
    def get_inputs(self):
        return [
            paddle.uniform([1738, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_32c7a5ee8018bbe57a501c88e04ddd0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c3b0bfbd1c337504fb396997e2c4582
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.1435141563415527]]], [[[1.5013723373413086]]], [[[0.8837188482284546]]], [[[1.2429099082946777]]], [[[1.0673458576202393]]], [[[1.6198890209197998]]], [[[1.6373324394226074]]], [[[1.2807600498199463]]], [[[1.7360910177230835]]], [[[1.3915798664093018]]], [[[1.262549638748169]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a9363ef46477617dbecc649fe1d81e7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c3b0bfbd1c337504fb396997e2c4582
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.3782894611358643]]], [[[1.0015931129455566]]], [[[1.0331555604934692]]], [[[1.9079935550689697]]], [[[1.6008999347686768]]], [[[1.641302227973938]]], [[[1.1618754863739014]]], [[[1.6235041618347168]]], [[[1.2821414470672607]]], [[[1.64640474319458]]], [[[1.8481693267822266]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_13f75fc3ab118dfff0dbcffbdd74f2a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c3b0bfbd1c337504fb396997e2c4582
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_024c73ca290019d704beac0de6fd610d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.floor(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6e8ee2bc1bd4b4eeb753ae889623fe2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_024c73ca290019d704beac0de6fd610d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_270fa595ef0167c80ed33e284bc83ec0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2b7f00520567e6075eb49f16358314c
    def get_inputs(self):
        return [
            paddle.uniform([5553, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_13f75fc3ab118dfff0dbcffbdd74f2a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c3b0bfbd1c337504fb396997e2c4582
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f75b316aeb5ca486116be39f697343ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2b7f00520567e6075eb49f16358314c
    def get_inputs(self):
        return [
            paddle.uniform([1733, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6411ba8b06cd8be9efd35bc0f6ca0aa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2b7f00520567e6075eb49f16358314c
    def get_inputs(self):
        return [
            paddle.uniform([1466, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a4cac8411d0a0fabd07238263fe0e874(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c3b0bfbd1c337504fb396997e2c4582
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.0663893222808838]]], [[[1.2209575176239014]]], [[[1.5715923309326172]]], [[[1.7096953392028809]]], [[[1.1510955095291138]]], [[[1.5153684616088867]]], [[[1.6526899337768555]]], [[[1.1250826120376587]]], [[[1.7514064311981201]]], [[[1.9096518754959106]]], [[[1.4441673755645752]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fbee89a1a50a862ba3195b0364ada8c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2b7f00520567e6075eb49f16358314c
    def get_inputs(self):
        return [
            paddle.uniform([2052, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_64d40dfb7a54941f54f1021df87150ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_024c73ca290019d704beac0de6fd610d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bdb6da45de40a18ec91d47d40700712d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2b7f00520567e6075eb49f16358314c
    def get_inputs(self):
        return [
            paddle.uniform([4717, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dc55b6a1b87c2e161b7b5dca015dfe6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2b7f00520567e6075eb49f16358314c
    def get_inputs(self):
        return [
            paddle.uniform([1056, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1032e3967f6a7a229b08d3b9453a6d70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_024c73ca290019d704beac0de6fd610d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8522008c7e3908924f932bbe9f5ac906(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2b7f00520567e6075eb49f16358314c
    def get_inputs(self):
        return [
            paddle.uniform([2354, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d0d3d52983ba50874c5bdc4696cda025(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2b7f00520567e6075eb49f16358314c
    def get_inputs(self):
        return [
            paddle.uniform([2994, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_22fdec8b11a22103af17ec59c151ea52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2b7f00520567e6075eb49f16358314c
    def get_inputs(self):
        return [
            paddle.uniform([3854, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_716c7d812e0ef47e03b4ddb7699dabf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_024c73ca290019d704beac0de6fd610d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_07622bc6c1b866b38fb17dfeec9441f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c3b0bfbd1c337504fb396997e2c4582
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.675636887550354]]], [[[1.4282872676849365]]], [[[1.901050329208374]]], [[[1.3266775608062744]]], [[[1.4924821853637695]]], [[[1.513107419013977]]], [[[1.1758352518081665]]], [[[0.9365093111991882]]], [[[1.161263108253479]]], [[[1.5383166074752808]]], [[[1.5137735605239868]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_13f75fc3ab118dfff0dbcffbdd74f2a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c3b0bfbd1c337504fb396997e2c4582
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dece30b2b0e461669bca2c591ee73d62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2b7f00520567e6075eb49f16358314c
    def get_inputs(self):
        return [
            paddle.uniform([2088, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_74911147a423a64302d5b3dcc232fe26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_024c73ca290019d704beac0de6fd610d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fe222245fe2659dd631c5860b9c141ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2b7f00520567e6075eb49f16358314c
    def get_inputs(self):
        return [
            paddle.uniform([4162, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_9be9eb7cfd49e0a468d6f413bacbbfa7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.floor(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_65b673c917e29dc3d525df2453b48544(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9be9eb7cfd49e0a468d6f413bacbbfa7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.9979315996170044]]], [[[1.4513835906982422]]], [[[1.4703072309494019]]], [[[1.514841914176941]]], [[[1.5136427879333496]]], [[[1.5811207294464111]]], [[[1.4758710861206055]]], [[[1.6939473152160645]]], [[[1.8386613130569458]]], [[[1.5455338954925537]]], [[[1.7069263458251953]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d870f6cc255a1ddf46b9316dca0d1651(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9be9eb7cfd49e0a468d6f413bacbbfa7
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d870f6cc255a1ddf46b9316dca0d1651(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9be9eb7cfd49e0a468d6f413bacbbfa7
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_9dc3d5d8d22b0a2e37c845170df0f25b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.floor(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a49fd3b103a4c106fd1f7867fb9b75ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9dc3d5d8d22b0a2e37c845170df0f25b
    def get_inputs(self):
        return [
            paddle.uniform([1738, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2315ce2b6732f9ae3e8a578dd723236d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9be9eb7cfd49e0a468d6f413bacbbfa7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.1435141563415527]]], [[[1.5013723373413086]]], [[[0.8837188482284546]]], [[[1.2429099082946777]]], [[[1.0673458576202393]]], [[[1.6198890209197998]]], [[[1.6373324394226074]]], [[[1.2807600498199463]]], [[[1.7360910177230835]]], [[[1.3915798664093018]]], [[[1.262549638748169]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9c84c85aa962ec68f57ee32785c0977d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9be9eb7cfd49e0a468d6f413bacbbfa7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.3782894611358643]]], [[[1.0015931129455566]]], [[[1.0331555604934692]]], [[[1.9079935550689697]]], [[[1.6008999347686768]]], [[[1.641302227973938]]], [[[1.1618754863739014]]], [[[1.6235041618347168]]], [[[1.2821414470672607]]], [[[1.64640474319458]]], [[[1.8481693267822266]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d870f6cc255a1ddf46b9316dca0d1651(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9be9eb7cfd49e0a468d6f413bacbbfa7
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_96e06f03ab10f6f1b772539b8b59300a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9be9eb7cfd49e0a468d6f413bacbbfa7
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_46eda4064a9acc848945e6f3d08db4ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9dc3d5d8d22b0a2e37c845170df0f25b
    def get_inputs(self):
        return [
            paddle.uniform([5553, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d870f6cc255a1ddf46b9316dca0d1651(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9be9eb7cfd49e0a468d6f413bacbbfa7
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0f4c2a26f3678723e72a3b10d923a1cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9dc3d5d8d22b0a2e37c845170df0f25b
    def get_inputs(self):
        return [
            paddle.uniform([1733, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_adc14f02ed438d31013ef1540dbfea63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9dc3d5d8d22b0a2e37c845170df0f25b
    def get_inputs(self):
        return [
            paddle.uniform([1466, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1ba542a758f8ef7aac189b82e7f37f35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9be9eb7cfd49e0a468d6f413bacbbfa7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.0663893222808838]]], [[[1.2209575176239014]]], [[[1.5715923309326172]]], [[[1.7096953392028809]]], [[[1.1510955095291138]]], [[[1.5153684616088867]]], [[[1.6526899337768555]]], [[[1.1250826120376587]]], [[[1.7514064311981201]]], [[[1.9096518754959106]]], [[[1.4441673755645752]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_76f2b9ced569e51977b5df7fbc43a3ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9dc3d5d8d22b0a2e37c845170df0f25b
    def get_inputs(self):
        return [
            paddle.uniform([2052, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ed9f1f193a9e96aed3f7462d5df3d07d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9be9eb7cfd49e0a468d6f413bacbbfa7
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_177372fefa8c6742a0385bcd134e26aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9dc3d5d8d22b0a2e37c845170df0f25b
    def get_inputs(self):
        return [
            paddle.uniform([4717, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6263c1a1030207bb90b8f473149bfda5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9dc3d5d8d22b0a2e37c845170df0f25b
    def get_inputs(self):
        return [
            paddle.uniform([1056, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_86123adc573ea00179f088fa6522993f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9be9eb7cfd49e0a468d6f413bacbbfa7
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dd4e97dceb96fc91da2aea1e68bd2e45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9dc3d5d8d22b0a2e37c845170df0f25b
    def get_inputs(self):
        return [
            paddle.uniform([2354, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_56be25ea73839a1a4477fc2d7b38d012(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9dc3d5d8d22b0a2e37c845170df0f25b
    def get_inputs(self):
        return [
            paddle.uniform([2994, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e4d9664a060966d1d84a09fdd366b734(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9dc3d5d8d22b0a2e37c845170df0f25b
    def get_inputs(self):
        return [
            paddle.uniform([3854, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1844d9a8bb60182a502be99f157b7790(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9be9eb7cfd49e0a468d6f413bacbbfa7
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0863ff09e269e36d18312821b38c7167(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9be9eb7cfd49e0a468d6f413bacbbfa7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.675636887550354]]], [[[1.4282872676849365]]], [[[1.901050329208374]]], [[[1.3266775608062744]]], [[[1.4924821853637695]]], [[[1.513107419013977]]], [[[1.1758352518081665]]], [[[0.9365093111991882]]], [[[1.161263108253479]]], [[[1.5383166074752808]]], [[[1.5137735605239868]]]], dtype='float32').reshape([11, 1, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d870f6cc255a1ddf46b9316dca0d1651(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9be9eb7cfd49e0a468d6f413bacbbfa7
    def get_inputs(self):
        return [
            paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cff3db3bf23af2d9c6ce1cdf66420646(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9dc3d5d8d22b0a2e37c845170df0f25b
    def get_inputs(self):
        return [
            paddle.uniform([2088, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c3f6f4ef39b3b4d2951ef700c0000f8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9be9eb7cfd49e0a468d6f413bacbbfa7
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_54fc118a10fe845cf73e3d646b319331(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9dc3d5d8d22b0a2e37c845170df0f25b
    def get_inputs(self):
        return [
            paddle.uniform([4162, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()



if __name__ == '__main__':
    unittest.main()