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
class PrimitiveOp_3855ae09e226a4ec347a43da72392937(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 2
        return paddle._C_ops.split_with_num(input_0, 2, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_213d4c3c6f5d5f6298467cd06666a8a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3855ae09e226a4ec347a43da72392937
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 4], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_8e9f91a899a1272fb9d07da656a34651(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 2
        return paddle._C_ops.split_with_num(input_0, 2, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dc819459b9f94fcd17fda453f9fd4b63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e9f91a899a1272fb9d07da656a34651
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d40ab2ae05adf1a260b336a778f28381(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3855ae09e226a4ec347a43da72392937
    def get_inputs(self):
        return [
            paddle.uniform([1, 12096, 4], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_940b1b492cc116cd16b43904671922a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 2
        return paddle._C_ops.split_with_num(input_0, 4, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3e32013d6498fb80f6ae753583ddba35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_940b1b492cc116cd16b43904671922a6
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 4, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a4bc322b1209df08c2d1653c319f4c0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e9f91a899a1272fb9d07da656a34651
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9f2be85fca3d49503d514e9cd28acfa3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e9f91a899a1272fb9d07da656a34651
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 4], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_d76522404a344b4ca0899124d3a4699a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1
        return paddle._C_ops.split_with_num(input_0, 2, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_85ee4206e9c8e9967c50e8616f7f160b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d76522404a344b4ca0899124d3a4699a
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 12, 12], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_00a64641c978dde8b4b439fff0eaa63b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1
        return paddle._C_ops.split_with_num(input_0, 4, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6b1e8a767f1e21c94ead35019428395b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00a64641c978dde8b4b439fff0eaa63b
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
class TestPrimitiveOp_6b1e8a767f1e21c94ead35019428395b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00a64641c978dde8b4b439fff0eaa63b
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
class TestPrimitiveOp_bc227f3fe0464afefbc487bdd5b1c948(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3855ae09e226a4ec347a43da72392937
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_0eec1517077824ef03868ea193c0d7ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d76522404a344b4ca0899124d3a4699a
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 8, 8], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_cbc59f563db28f7ce1a2dba7b5683078(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e9f91a899a1272fb9d07da656a34651
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c413d1a9591aa65433a43388e848ab87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3855ae09e226a4ec347a43da72392937
    def get_inputs(self):
        return [
            paddle.uniform([1, 5376, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_dc819459b9f94fcd17fda453f9fd4b63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e9f91a899a1272fb9d07da656a34651
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_9fb3f50969088892354b4cd480b4b65c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1
        return paddle._C_ops.split_with_num(input_0, 8, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e47dc4f6d51b59ec88495cc908a1483e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9fb3f50969088892354b4cd480b4b65c
    def get_inputs(self):
        return [
            paddle.uniform([22, 224, 28, 28], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_354d22fe7989fc93c5ca7c6e2254a9f8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1
        return paddle._C_ops.split_with_num(input_0, 4, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1248d5c5f4da630996fc30bd40031ef0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_354d22fe7989fc93c5ca7c6e2254a9f8
    def get_inputs(self):
        return [
            paddle.uniform([9, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1248d5c5f4da630996fc30bd40031ef0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_354d22fe7989fc93c5ca7c6e2254a9f8
    def get_inputs(self):
        return [
            paddle.uniform([9, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8732eb2595df9a92369434edfafc0cd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d76522404a344b4ca0899124d3a4699a
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 128, 256], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_fa320f37af718dd6f11b5fbe310a5be0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d76522404a344b4ca0899124d3a4699a
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 64, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8fce62e830c3983b1f34b7c2c9f83275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d76522404a344b4ca0899124d3a4699a
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 32, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_459a08203065cefe3b7cee9cba46c727(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d76522404a344b4ca0899124d3a4699a
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 16, 32], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_2ee90b7f887d65224805048ffbf6acd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00a64641c978dde8b4b439fff0eaa63b
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
class TestPrimitiveOp_2ee90b7f887d65224805048ffbf6acd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00a64641c978dde8b4b439fff0eaa63b
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
class TestPrimitiveOp_4ed0baaf8475f97aec5cd091d9f11a49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3855ae09e226a4ec347a43da72392937
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_81d664def4d6fb63e2a3b6f3b18edf0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9fb3f50969088892354b4cd480b4b65c
    def get_inputs(self):
        return [
            paddle.uniform([22, 224, 56, 56], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_50dc0dd37d5593eaf75f37b319cd1f2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9fb3f50969088892354b4cd480b4b65c
    def get_inputs(self):
        return [
            paddle.uniform([10, 224, 56, 56], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_4bd8e9da750bdbb92f75de2ba37bdf8a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0
        return paddle._C_ops.split_with_num(input_0, 4, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e5a001d970b0b77d00ed3f7a9fbc5183(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bd8e9da750bdbb92f75de2ba37bdf8a
    def get_inputs(self):
        return [
            paddle.to_tensor([0.48949873447418213, 0.2125146985054016, 0.3943987786769867, 0.052135169506073, 0.3060843348503113, 0.14483611285686493, 0.1455189287662506, 0.18382209539413452, 0.022990306839346886, 0.4498473107814789, 0.47428104281425476, 0.17863892018795013, 0.033982645720243454, 0.30670925974845886, 0.2704288363456726, 0.40632981061935425, 0.399911105632782, 0.4827656149864197, 0.46267789602279663, 0.06502489745616913, 0.23793815076351166, 0.4408537447452545, 0.28138476610183716, 0.4201972782611847], dtype='float32').reshape([24]),
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
class TestPrimitiveOp_3dfa1b0f10afbcf09cca9a8da02d5c34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bd8e9da750bdbb92f75de2ba37bdf8a
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3529529273509979, 0.1577829271554947, 0.3188677430152893, 0.33222490549087524, 0.2642192542552948, 0.17672815918922424, 0.2904326915740967, 0.35068219900131226, 0.13571391999721527, 0.24345217645168304, 0.447722464799881, 0.3805902302265167, 0.4372062087059021, 0.4139530062675476, 0.03780849650502205, 0.035732731223106384, 0.03976374492049217, 0.21069513261318207, 0.28935226798057556, 0.455740362405777, 0.22648458182811737, 0.2802315354347229, 0.46797147393226624, 0.46781590580940247], dtype='float32').reshape([24]),
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
class TestPrimitiveOp_3054c7ea274468e23c176dad19088e32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00a64641c978dde8b4b439fff0eaa63b
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
class TestPrimitiveOp_3054c7ea274468e23c176dad19088e32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00a64641c978dde8b4b439fff0eaa63b
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
class TestPrimitiveOp_bc227f3fe0464afefbc487bdd5b1c948(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3855ae09e226a4ec347a43da72392937
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_eeceb60626ae624c8cb1a0f4ab4d35fc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 2
        return paddle._C_ops.split_with_num(input_0, 3, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 2304], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fde0ab9978c8bee3cbde20ea286b7c66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eeceb60626ae624c8cb1a0f4ab4d35fc
    def get_inputs(self):
        return [
            paddle.uniform([4, 144, 2304], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8732eb2595df9a92369434edfafc0cd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d76522404a344b4ca0899124d3a4699a
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 128, 256], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_fa320f37af718dd6f11b5fbe310a5be0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d76522404a344b4ca0899124d3a4699a
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 64, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8fce62e830c3983b1f34b7c2c9f83275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d76522404a344b4ca0899124d3a4699a
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 32, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_062df11adacb2061d81e3b194b2f1732(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e9f91a899a1272fb9d07da656a34651
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5a53ab8ae209038a7ce80748331842fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d76522404a344b4ca0899124d3a4699a
    def get_inputs(self):
        return [
            paddle.uniform([1, 34, 128, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_916f3bde9b929c655a2b886faa95d9b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3855ae09e226a4ec347a43da72392937
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_220546f0d452cee43f727283d7013a93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9fb3f50969088892354b4cd480b4b65c
    def get_inputs(self):
        return [
            paddle.uniform([22, 896, 14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b0bd6c5b9b0b7d57f5f722cfca97f912(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e9f91a899a1272fb9d07da656a34651
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a6c0bf5e53301e37dc0fd16efa483285(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00a64641c978dde8b4b439fff0eaa63b
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
class TestPrimitiveOp_a6c0bf5e53301e37dc0fd16efa483285(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00a64641c978dde8b4b439fff0eaa63b
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
class TestPrimitiveOp_640e02dc0469ba0f238c9921d5e2ba56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3855ae09e226a4ec347a43da72392937
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_0c1db86eb4d9bc48b368db383bfa0568(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9fb3f50969088892354b4cd480b4b65c
    def get_inputs(self):
        return [
            paddle.uniform([22, 448, 28, 28], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b892c037775a1a6015b75e5d36343b27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9fb3f50969088892354b4cd480b4b65c
    def get_inputs(self):
        return [
            paddle.uniform([10, 112, 56, 56], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b68134f98c1def37b1b54f5a4bb97472(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_354d22fe7989fc93c5ca7c6e2254a9f8
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.07675330340862274, 0.04051472991704941, 0.2655336260795593, 0.06180190667510033]], dtype='float32').reshape([1, 4]),
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
class TestPrimitiveOp_dcaeb872f8133ea8774cbd969d34ca65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_354d22fe7989fc93c5ca7c6e2254a9f8
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.399107426404953, 0.11604505032300949, 0.2398943156003952, 0.368866503238678]], dtype='float32').reshape([1, 4]),
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
class TestPrimitiveOp_67faec5e4eae6a818c05b2ea9e5e3c31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_354d22fe7989fc93c5ca7c6e2254a9f8
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4892083704471588, 0.21311093866825104, 0.3479418456554413, 0.09062915295362473], [0.27091413736343384, 0.05007264018058777, 0.34206390380859375, 0.18488666415214539], [0.17127905786037445, 0.38398033380508423, 0.08347898721694946, 0.34276267886161804], [0.1577376276254654, 0.09370165318250656, 0.061992086470127106, 0.2707947790622711], [0.41942358016967773, 0.32851335406303406, 0.2510312795639038, 0.3789082169532776], [0.17888130247592926, 0.1493993103504181, 0.4961717128753662, 0.28018254041671753]], dtype='float32').reshape([6, 4]),
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
class TestPrimitiveOp_623166c271476ad9b0cd68e823e2ace9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_354d22fe7989fc93c5ca7c6e2254a9f8
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06633955240249634, 0.37008339166641235, 0.49149706959724426, 0.07251711189746857], [0.18247836828231812, 0.3361941874027252, 0.3978572487831116, 0.2581758201122284], [0.2487204223871231, 0.14341436326503754, 0.2893059551715851, 0.3393520414829254], [0.44636279344558716, 0.14076094329357147, 0.04706785827875137, 0.4828045070171356], [0.2324966937303543, 0.3007638156414032, 0.09867657721042633, 0.3792950510978699], [0.3986716568470001, 0.4232746660709381, 0.3818233013153076, 0.3344321846961975]], dtype='float32').reshape([6, 4]),
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


class PrimitiveOp_5ebac0042bb380ad0b76d69c8e72263c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 2
        return paddle._C_ops.split_with_num(input_0, 3, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 1152], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_49234c6b91818fa6af9bc65443ca936b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ebac0042bb380ad0b76d69c8e72263c
    def get_inputs(self):
        return [
            paddle.uniform([4, 576, 1152], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_4ffeeb3561af8cc75a46ea9f3b8e4d35(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 2
        return paddle._C_ops.split_with_num(input_0, 3, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 576], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d36b4b2d49244bfccb9d58ec15cbb5c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ffeeb3561af8cc75a46ea9f3b8e4d35
    def get_inputs(self):
        return [
            paddle.uniform([4, 2304, 576], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_918ca38d32dda60dc1c915f64bc9b30f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00a64641c978dde8b4b439fff0eaa63b
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
class TestPrimitiveOp_918ca38d32dda60dc1c915f64bc9b30f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00a64641c978dde8b4b439fff0eaa63b
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
class TestPrimitiveOp_e4c260a5194a715316628376a1608555(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3855ae09e226a4ec347a43da72392937
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_90469cf5ebe864636a5bf6f0530721e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9fb3f50969088892354b4cd480b4b65c
    def get_inputs(self):
        return [
            paddle.uniform([22, 112, 56, 56], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_78dfe97ee2612fd0ef8d86cf232f9d07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00a64641c978dde8b4b439fff0eaa63b
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
class TestPrimitiveOp_78dfe97ee2612fd0ef8d86cf232f9d07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00a64641c978dde8b4b439fff0eaa63b
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
class TestPrimitiveOp_df8a6611580c0f87a39509df6757bb80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3855ae09e226a4ec347a43da72392937
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c2ed2888e2f0e037e36cf2fe2e7cdefe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d76522404a344b4ca0899124d3a4699a
    def get_inputs(self):
        return [
            paddle.uniform([1, 34, 160, 160], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_e429ad926d23e977d39b7ce7ed63493f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00a64641c978dde8b4b439fff0eaa63b
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
class TestPrimitiveOp_e429ad926d23e977d39b7ce7ed63493f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00a64641c978dde8b4b439fff0eaa63b
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
class TestPrimitiveOp_62a98d04cf11ef39d4b091d0279eab00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3855ae09e226a4ec347a43da72392937
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8565cd52817791204ae182f0a76fab1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e9f91a899a1272fb9d07da656a34651
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_3cf6239af54670274a13719c439bbdc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9fb3f50969088892354b4cd480b4b65c
    def get_inputs(self):
        return [
            paddle.uniform([22, 448, 14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_bb15635a0bdd29ced4fe1288d760153c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eeceb60626ae624c8cb1a0f4ab4d35fc
    def get_inputs(self):
        return [
            paddle.uniform([6, 144, 2304], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f03c542dad1368bc2649506aa10f21e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3855ae09e226a4ec347a43da72392937
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a7e255c9615270eef053d6fc0eb96840(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_354d22fe7989fc93c5ca7c6e2254a9f8
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.121558278799057, 0.06055105850100517, 0.23420564830303192, 0.12737542390823364], [0.026957757771015167, 0.3023223578929901, 0.1628895252943039, 0.2753457725048065], [0.3154665231704712, 0.29442358016967773, 0.17040301859378815, 0.42593345046043396], [0.13702236115932465, 0.12275972962379456, 0.04161243140697479, 0.11517052352428436], [0.13035985827445984, 0.3305083215236664, 0.31444212794303894, 0.06334847956895828]], dtype='float32').reshape([5, 4]),
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
class TestPrimitiveOp_e7d3d740cbc1377ddfb2dec170e006ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_354d22fe7989fc93c5ca7c6e2254a9f8
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.29517462849617004, 0.015375080518424511, 0.20128846168518066, 0.2253778576850891], [0.09762095659971237, 0.07627338171005249, 0.1977842003107071, 0.03415966406464577], [0.39388298988342285, 0.2160526067018509, 0.1439625471830368, 0.43615180253982544], [0.45222628116607666, 0.0893891453742981, 0.07181569188833237, 0.47151103615760803], [0.37068048119544983, 0.40267425775527954, 0.2501826286315918, 0.14356163144111633]], dtype='float32').reshape([5, 4]),
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
class TestPrimitiveOp_71bcf2ffbe9d4249f67549359b554fcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e9f91a899a1272fb9d07da656a34651
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 4], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_96d28b0cb8da641793c8210e0da534c6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 2
        return paddle._C_ops.split_with_num(input_0, 3, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 288], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_56306ef533dae98419a9dc72b5062b08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96d28b0cb8da641793c8210e0da534c6
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 288], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ca79bbe0deae30db33cff6fece6d9f90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d76522404a344b4ca0899124d3a4699a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b09ecb1a92d78785998e25a14a7cf796(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9fb3f50969088892354b4cd480b4b65c
    def get_inputs(self):
        return [
            paddle.uniform([22, 896, 7, 7], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c8646a06cbb98ce08fb1a233979e71e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00a64641c978dde8b4b439fff0eaa63b
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
class TestPrimitiveOp_c8646a06cbb98ce08fb1a233979e71e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00a64641c978dde8b4b439fff0eaa63b
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
class TestPrimitiveOp_502a478c443cb8099992d99f8b7960c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3855ae09e226a4ec347a43da72392937
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1a61854bf433aa253952aa4b9cf0f829(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00a64641c978dde8b4b439fff0eaa63b
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
class TestPrimitiveOp_1a61854bf433aa253952aa4b9cf0f829(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00a64641c978dde8b4b439fff0eaa63b
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
class TestPrimitiveOp_f03c542dad1368bc2649506aa10f21e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3855ae09e226a4ec347a43da72392937
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_79a83da7b867457a39a6397550a304da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00a64641c978dde8b4b439fff0eaa63b
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
class TestPrimitiveOp_79a83da7b867457a39a6397550a304da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00a64641c978dde8b4b439fff0eaa63b
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
class TestPrimitiveOp_3b983f5cd1abdd4ded89d6072fa91ae9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3855ae09e226a4ec347a43da72392937
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_86e644ee43855b1323ab5ab67b2da592(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e9f91a899a1272fb9d07da656a34651
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_085e7b992dfdc62a375905b77c3655fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96d28b0cb8da641793c8210e0da534c6
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 288], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f8462382d24f6ca281aa331674ce7c4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ffeeb3561af8cc75a46ea9f3b8e4d35
    def get_inputs(self):
        return [
            paddle.uniform([6, 2304, 576], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a17196478d9a3ce9b1f8f37d8cb27434(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_354d22fe7989fc93c5ca7c6e2254a9f8
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.30841371417045593, 0.05442650616168976, 0.2406461387872696, 0.48310473561286926], [0.16765755414962769, 0.35331493616104126, 0.24191485345363617, 0.2981303632259369], [0.24464209377765656, 0.08072906732559204, 0.32453492283821106, 0.07768220454454422], [0.4673091173171997, 0.19166995584964752, 0.31000232696533203, 0.4662059247493744]], dtype='float32').reshape([4, 4]),
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
class TestPrimitiveOp_321edf1b059935d433c3b38224c4a49c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_354d22fe7989fc93c5ca7c6e2254a9f8
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.08165464550256729, 0.20266421139240265, 0.2287028282880783, 0.029762178659439087], [0.20990590751171112, 0.19550111889839172, 0.38752099871635437, 0.26578471064567566], [0.34994369745254517, 0.2868219316005707, 0.013475647196173668, 0.1908850222826004], [0.2936878800392151, 0.05105381831526756, 0.4039143919944763, 0.07886908948421478]], dtype='float32').reshape([4, 4]),
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
class TestPrimitiveOp_c87fa07a6aee6ebcca306247f68a8a4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ebac0042bb380ad0b76d69c8e72263c
    def get_inputs(self):
        return [
            paddle.uniform([6, 576, 1152], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_2c1a0bf8387784c8385d849104259737(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00a64641c978dde8b4b439fff0eaa63b
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
class TestPrimitiveOp_2c1a0bf8387784c8385d849104259737(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00a64641c978dde8b4b439fff0eaa63b
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
class TestPrimitiveOp_e4c260a5194a715316628376a1608555(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3855ae09e226a4ec347a43da72392937
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_063094c25a738c67634239eaca72fe53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e9f91a899a1272fb9d07da656a34651
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_96738eb73620d0c74b5c43552013c8fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3855ae09e226a4ec347a43da72392937
    def get_inputs(self):
        return [
            paddle.uniform([1, 6804, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_52b3809541afd9c5dae207b8b341fe9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00a64641c978dde8b4b439fff0eaa63b
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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_52b3809541afd9c5dae207b8b341fe9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00a64641c978dde8b4b439fff0eaa63b
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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_916f3bde9b929c655a2b886faa95d9b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3855ae09e226a4ec347a43da72392937
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_213d4c3c6f5d5f6298467cd06666a8a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3855ae09e226a4ec347a43da72392937
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_bc227f3fe0464afefbc487bdd5b1c948(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3855ae09e226a4ec347a43da72392937
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d40ab2ae05adf1a260b336a778f28381(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3855ae09e226a4ec347a43da72392937
    def get_inputs(self):
        return [
            paddle.uniform([1, 12096, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_3e32013d6498fb80f6ae753583ddba35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_940b1b492cc116cd16b43904671922a6
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 4, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_3b983f5cd1abdd4ded89d6072fa91ae9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3855ae09e226a4ec347a43da72392937
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_502a478c443cb8099992d99f8b7960c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3855ae09e226a4ec347a43da72392937
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_85ee4206e9c8e9967c50e8616f7f160b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d76522404a344b4ca0899124d3a4699a
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b14bb7fdfebca9d65b123f151ff2a3fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_354d22fe7989fc93c5ca7c6e2254a9f8
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
class TestPrimitiveOp_b14bb7fdfebca9d65b123f151ff2a3fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_354d22fe7989fc93c5ca7c6e2254a9f8
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
class TestPrimitiveOp_bc227f3fe0464afefbc487bdd5b1c948(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3855ae09e226a4ec347a43da72392937
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_0eec1517077824ef03868ea193c0d7ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d76522404a344b4ca0899124d3a4699a
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 8, 8], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_916f3bde9b929c655a2b886faa95d9b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3855ae09e226a4ec347a43da72392937
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c413d1a9591aa65433a43388e848ab87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3855ae09e226a4ec347a43da72392937
    def get_inputs(self):
        return [
            paddle.uniform([1, 5376, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_bc227f3fe0464afefbc487bdd5b1c948(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3855ae09e226a4ec347a43da72392937
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_e47dc4f6d51b59ec88495cc908a1483e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9fb3f50969088892354b4cd480b4b65c
    def get_inputs(self):
        return [
            paddle.uniform([22, 224, 28, 28], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1248d5c5f4da630996fc30bd40031ef0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_354d22fe7989fc93c5ca7c6e2254a9f8
    def get_inputs(self):
        return [
            paddle.uniform([9, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1248d5c5f4da630996fc30bd40031ef0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_354d22fe7989fc93c5ca7c6e2254a9f8
    def get_inputs(self):
        return [
            paddle.uniform([9, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8732eb2595df9a92369434edfafc0cd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d76522404a344b4ca0899124d3a4699a
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 128, 256], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_fa320f37af718dd6f11b5fbe310a5be0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d76522404a344b4ca0899124d3a4699a
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 64, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8fce62e830c3983b1f34b7c2c9f83275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d76522404a344b4ca0899124d3a4699a
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 32, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_459a08203065cefe3b7cee9cba46c727(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d76522404a344b4ca0899124d3a4699a
    def get_inputs(self):
        return [
            paddle.uniform([1, 320, 16, 32], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_07182800521430d7f53c1443d130adfe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_354d22fe7989fc93c5ca7c6e2254a9f8
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
class TestPrimitiveOp_07182800521430d7f53c1443d130adfe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_354d22fe7989fc93c5ca7c6e2254a9f8
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
class TestPrimitiveOp_4ed0baaf8475f97aec5cd091d9f11a49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3855ae09e226a4ec347a43da72392937
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_81d664def4d6fb63e2a3b6f3b18edf0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9fb3f50969088892354b4cd480b4b65c
    def get_inputs(self):
        return [
            paddle.uniform([22, 224, 56, 56], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_50dc0dd37d5593eaf75f37b319cd1f2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9fb3f50969088892354b4cd480b4b65c
    def get_inputs(self):
        return [
            paddle.uniform([10, 224, 56, 56], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_e5a001d970b0b77d00ed3f7a9fbc5183(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bd8e9da750bdbb92f75de2ba37bdf8a
    def get_inputs(self):
        return [
            paddle.to_tensor([0.48949873447418213, 0.2125146985054016, 0.3943987786769867, 0.052135169506073, 0.3060843348503113, 0.14483611285686493, 0.1455189287662506, 0.18382209539413452, 0.022990306839346886, 0.4498473107814789, 0.47428104281425476, 0.17863892018795013, 0.033982645720243454, 0.30670925974845886, 0.2704288363456726, 0.40632981061935425, 0.399911105632782, 0.4827656149864197, 0.46267789602279663, 0.06502489745616913, 0.23793815076351166, 0.4408537447452545, 0.28138476610183716, 0.4201972782611847], dtype='float32').reshape([24]),
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
class TestPrimitiveOp_3dfa1b0f10afbcf09cca9a8da02d5c34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bd8e9da750bdbb92f75de2ba37bdf8a
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3529529273509979, 0.1577829271554947, 0.3188677430152893, 0.33222490549087524, 0.2642192542552948, 0.17672815918922424, 0.2904326915740967, 0.35068219900131226, 0.13571391999721527, 0.24345217645168304, 0.447722464799881, 0.3805902302265167, 0.4372062087059021, 0.4139530062675476, 0.03780849650502205, 0.035732731223106384, 0.03976374492049217, 0.21069513261318207, 0.28935226798057556, 0.455740362405777, 0.22648458182811737, 0.2802315354347229, 0.46797147393226624, 0.46781590580940247], dtype='float32').reshape([24]),
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
class TestPrimitiveOp_6e02cd12b2933deeb7f1652eda9fd427(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_354d22fe7989fc93c5ca7c6e2254a9f8
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
class TestPrimitiveOp_6e02cd12b2933deeb7f1652eda9fd427(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_354d22fe7989fc93c5ca7c6e2254a9f8
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
class TestPrimitiveOp_bc227f3fe0464afefbc487bdd5b1c948(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3855ae09e226a4ec347a43da72392937
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_98e2ac6b6414c30a52779b61098e26ee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 2
        return paddle._C_ops.split_with_num(input_0, 3, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a443d1ce158d2bb68df9bb36cf8a9e35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98e2ac6b6414c30a52779b61098e26ee
    def get_inputs(self):
        return [
            paddle.uniform([4, 144, 2304], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8732eb2595df9a92369434edfafc0cd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d76522404a344b4ca0899124d3a4699a
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 128, 256], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_fa320f37af718dd6f11b5fbe310a5be0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d76522404a344b4ca0899124d3a4699a
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 64, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8fce62e830c3983b1f34b7c2c9f83275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d76522404a344b4ca0899124d3a4699a
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 32, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_e4c260a5194a715316628376a1608555(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3855ae09e226a4ec347a43da72392937
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5a53ab8ae209038a7ce80748331842fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d76522404a344b4ca0899124d3a4699a
    def get_inputs(self):
        return [
            paddle.uniform([1, 34, 128, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_916f3bde9b929c655a2b886faa95d9b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3855ae09e226a4ec347a43da72392937
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_220546f0d452cee43f727283d7013a93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9fb3f50969088892354b4cd480b4b65c
    def get_inputs(self):
        return [
            paddle.uniform([22, 896, 14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f03c542dad1368bc2649506aa10f21e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3855ae09e226a4ec347a43da72392937
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_81230d87897f45ce01cb19b646f3355c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_354d22fe7989fc93c5ca7c6e2254a9f8
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
class TestPrimitiveOp_81230d87897f45ce01cb19b646f3355c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_354d22fe7989fc93c5ca7c6e2254a9f8
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
class TestPrimitiveOp_640e02dc0469ba0f238c9921d5e2ba56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3855ae09e226a4ec347a43da72392937
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_0c1db86eb4d9bc48b368db383bfa0568(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9fb3f50969088892354b4cd480b4b65c
    def get_inputs(self):
        return [
            paddle.uniform([22, 448, 28, 28], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b892c037775a1a6015b75e5d36343b27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9fb3f50969088892354b4cd480b4b65c
    def get_inputs(self):
        return [
            paddle.uniform([10, 112, 56, 56], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b68134f98c1def37b1b54f5a4bb97472(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_354d22fe7989fc93c5ca7c6e2254a9f8
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.07675330340862274, 0.04051472991704941, 0.2655336260795593, 0.06180190667510033]], dtype='float32').reshape([1, 4]),
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
class TestPrimitiveOp_dcaeb872f8133ea8774cbd969d34ca65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_354d22fe7989fc93c5ca7c6e2254a9f8
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.399107426404953, 0.11604505032300949, 0.2398943156003952, 0.368866503238678]], dtype='float32').reshape([1, 4]),
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
class TestPrimitiveOp_67faec5e4eae6a818c05b2ea9e5e3c31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_354d22fe7989fc93c5ca7c6e2254a9f8
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4892083704471588, 0.21311093866825104, 0.3479418456554413, 0.09062915295362473], [0.27091413736343384, 0.05007264018058777, 0.34206390380859375, 0.18488666415214539], [0.17127905786037445, 0.38398033380508423, 0.08347898721694946, 0.34276267886161804], [0.1577376276254654, 0.09370165318250656, 0.061992086470127106, 0.2707947790622711], [0.41942358016967773, 0.32851335406303406, 0.2510312795639038, 0.3789082169532776], [0.17888130247592926, 0.1493993103504181, 0.4961717128753662, 0.28018254041671753]], dtype='float32').reshape([6, 4]),
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
class TestPrimitiveOp_623166c271476ad9b0cd68e823e2ace9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_354d22fe7989fc93c5ca7c6e2254a9f8
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06633955240249634, 0.37008339166641235, 0.49149706959724426, 0.07251711189746857], [0.18247836828231812, 0.3361941874027252, 0.3978572487831116, 0.2581758201122284], [0.2487204223871231, 0.14341436326503754, 0.2893059551715851, 0.3393520414829254], [0.44636279344558716, 0.14076094329357147, 0.04706785827875137, 0.4828045070171356], [0.2324966937303543, 0.3007638156414032, 0.09867657721042633, 0.3792950510978699], [0.3986716568470001, 0.4232746660709381, 0.3818233013153076, 0.3344321846961975]], dtype='float32').reshape([6, 4]),
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
class TestPrimitiveOp_8fe357b8d048ab12d480ec5cbbe673ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98e2ac6b6414c30a52779b61098e26ee
    def get_inputs(self):
        return [
            paddle.uniform([4, 576, 1152], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a2889a31ff592c0fdf8ba42fe01eb4d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98e2ac6b6414c30a52779b61098e26ee
    def get_inputs(self):
        return [
            paddle.uniform([4, 2304, 576], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_2235d8144ac803fe37ed0e2c2a33bbb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_354d22fe7989fc93c5ca7c6e2254a9f8
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
class TestPrimitiveOp_2235d8144ac803fe37ed0e2c2a33bbb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_354d22fe7989fc93c5ca7c6e2254a9f8
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
class TestPrimitiveOp_e4c260a5194a715316628376a1608555(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3855ae09e226a4ec347a43da72392937
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_90469cf5ebe864636a5bf6f0530721e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9fb3f50969088892354b4cd480b4b65c
    def get_inputs(self):
        return [
            paddle.uniform([22, 112, 56, 56], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_72407ac394cd65160b62f54079adeff8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_354d22fe7989fc93c5ca7c6e2254a9f8
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
class TestPrimitiveOp_72407ac394cd65160b62f54079adeff8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_354d22fe7989fc93c5ca7c6e2254a9f8
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
class TestPrimitiveOp_df8a6611580c0f87a39509df6757bb80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3855ae09e226a4ec347a43da72392937
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c2ed2888e2f0e037e36cf2fe2e7cdefe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d76522404a344b4ca0899124d3a4699a
    def get_inputs(self):
        return [
            paddle.uniform([1, 34, 160, 160], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_0c24b43bbc426a5f566448334d6a9353(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_354d22fe7989fc93c5ca7c6e2254a9f8
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
class TestPrimitiveOp_0c24b43bbc426a5f566448334d6a9353(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_354d22fe7989fc93c5ca7c6e2254a9f8
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
class TestPrimitiveOp_62a98d04cf11ef39d4b091d0279eab00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3855ae09e226a4ec347a43da72392937
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_df8a6611580c0f87a39509df6757bb80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3855ae09e226a4ec347a43da72392937
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_3cf6239af54670274a13719c439bbdc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9fb3f50969088892354b4cd480b4b65c
    def get_inputs(self):
        return [
            paddle.uniform([22, 448, 14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c60319ebc5898abd57a40850521c09b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98e2ac6b6414c30a52779b61098e26ee
    def get_inputs(self):
        return [
            paddle.uniform([6, 144, 2304], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f03c542dad1368bc2649506aa10f21e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3855ae09e226a4ec347a43da72392937
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a7e255c9615270eef053d6fc0eb96840(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_354d22fe7989fc93c5ca7c6e2254a9f8
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.121558278799057, 0.06055105850100517, 0.23420564830303192, 0.12737542390823364], [0.026957757771015167, 0.3023223578929901, 0.1628895252943039, 0.2753457725048065], [0.3154665231704712, 0.29442358016967773, 0.17040301859378815, 0.42593345046043396], [0.13702236115932465, 0.12275972962379456, 0.04161243140697479, 0.11517052352428436], [0.13035985827445984, 0.3305083215236664, 0.31444212794303894, 0.06334847956895828]], dtype='float32').reshape([5, 4]),
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
class TestPrimitiveOp_e7d3d740cbc1377ddfb2dec170e006ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_354d22fe7989fc93c5ca7c6e2254a9f8
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.29517462849617004, 0.015375080518424511, 0.20128846168518066, 0.2253778576850891], [0.09762095659971237, 0.07627338171005249, 0.1977842003107071, 0.03415966406464577], [0.39388298988342285, 0.2160526067018509, 0.1439625471830368, 0.43615180253982544], [0.45222628116607666, 0.0893891453742981, 0.07181569188833237, 0.47151103615760803], [0.37068048119544983, 0.40267425775527954, 0.2501826286315918, 0.14356163144111633]], dtype='float32').reshape([5, 4]),
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
class TestPrimitiveOp_62a98d04cf11ef39d4b091d0279eab00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3855ae09e226a4ec347a43da72392937
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_24af8c40480bb108e81d470680a98569(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98e2ac6b6414c30a52779b61098e26ee
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 288], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ca79bbe0deae30db33cff6fece6d9f90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d76522404a344b4ca0899124d3a4699a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b09ecb1a92d78785998e25a14a7cf796(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9fb3f50969088892354b4cd480b4b65c
    def get_inputs(self):
        return [
            paddle.uniform([22, 896, 7, 7], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_94484901898be6a9416d81c5dbaa041c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_354d22fe7989fc93c5ca7c6e2254a9f8
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
class TestPrimitiveOp_94484901898be6a9416d81c5dbaa041c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_354d22fe7989fc93c5ca7c6e2254a9f8
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
class TestPrimitiveOp_502a478c443cb8099992d99f8b7960c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3855ae09e226a4ec347a43da72392937
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_2643d108a226e70aa00c14eaffe9c1e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_354d22fe7989fc93c5ca7c6e2254a9f8
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
class TestPrimitiveOp_2643d108a226e70aa00c14eaffe9c1e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_354d22fe7989fc93c5ca7c6e2254a9f8
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
class TestPrimitiveOp_f03c542dad1368bc2649506aa10f21e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3855ae09e226a4ec347a43da72392937
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_0cc5d6278bd96553f469507519252a85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_354d22fe7989fc93c5ca7c6e2254a9f8
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
class TestPrimitiveOp_0cc5d6278bd96553f469507519252a85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_354d22fe7989fc93c5ca7c6e2254a9f8
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
class TestPrimitiveOp_3b983f5cd1abdd4ded89d6072fa91ae9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3855ae09e226a4ec347a43da72392937
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4ed0baaf8475f97aec5cd091d9f11a49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3855ae09e226a4ec347a43da72392937
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d4aaf6f15b69de76be93bb4a54109844(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98e2ac6b6414c30a52779b61098e26ee
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 288], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6b5780110046dd209e1cd685a3cf00ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98e2ac6b6414c30a52779b61098e26ee
    def get_inputs(self):
        return [
            paddle.uniform([6, 2304, 576], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a17196478d9a3ce9b1f8f37d8cb27434(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_354d22fe7989fc93c5ca7c6e2254a9f8
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.30841371417045593, 0.05442650616168976, 0.2406461387872696, 0.48310473561286926], [0.16765755414962769, 0.35331493616104126, 0.24191485345363617, 0.2981303632259369], [0.24464209377765656, 0.08072906732559204, 0.32453492283821106, 0.07768220454454422], [0.4673091173171997, 0.19166995584964752, 0.31000232696533203, 0.4662059247493744]], dtype='float32').reshape([4, 4]),
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
class TestPrimitiveOp_321edf1b059935d433c3b38224c4a49c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_354d22fe7989fc93c5ca7c6e2254a9f8
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.08165464550256729, 0.20266421139240265, 0.2287028282880783, 0.029762178659439087], [0.20990590751171112, 0.19550111889839172, 0.38752099871635437, 0.26578471064567566], [0.34994369745254517, 0.2868219316005707, 0.013475647196173668, 0.1908850222826004], [0.2936878800392151, 0.05105381831526756, 0.4039143919944763, 0.07886908948421478]], dtype='float32').reshape([4, 4]),
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
class TestPrimitiveOp_88352daebf921088ff5319c5a279f4b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98e2ac6b6414c30a52779b61098e26ee
    def get_inputs(self):
        return [
            paddle.uniform([6, 576, 1152], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_700a06c8e009c15c098335ff53e39dc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_354d22fe7989fc93c5ca7c6e2254a9f8
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
class TestPrimitiveOp_700a06c8e009c15c098335ff53e39dc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_354d22fe7989fc93c5ca7c6e2254a9f8
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
class TestPrimitiveOp_e4c260a5194a715316628376a1608555(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3855ae09e226a4ec347a43da72392937
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_640e02dc0469ba0f238c9921d5e2ba56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3855ae09e226a4ec347a43da72392937
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_96738eb73620d0c74b5c43552013c8fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3855ae09e226a4ec347a43da72392937
    def get_inputs(self):
        return [
            paddle.uniform([1, 6804, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_217f6a78fb01bc040f057c65775c99ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_354d22fe7989fc93c5ca7c6e2254a9f8
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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_217f6a78fb01bc040f057c65775c99ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_354d22fe7989fc93c5ca7c6e2254a9f8
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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_916f3bde9b929c655a2b886faa95d9b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3855ae09e226a4ec347a43da72392937
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 4], dtype='float32', min=0, max=0.5),
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