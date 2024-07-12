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
class PrimitiveOp_19d0a8a77ea8ac0dd0fb5dcf3895371d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0b42cee1f3c6b7f1a3c289d5c586e1b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19d0a8a77ea8ac0dd0fb5dcf3895371d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.3018275201320648]], [[0.33168932795524597]], [[0.46974313259124756]], [[0.010415264405310154]], [[0.16770502924919128]], [[0.08946999907493591]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.5554413795471191]], [[0.7320908904075623]], [[0.6382142901420593]], [[0.7114271521568298]], [[0.7790687680244446]], [[0.804796040058136]]], dtype='float32').reshape([6, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_84dbe9af23782d6554cfd945c82d3b06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19d0a8a77ea8ac0dd0fb5dcf3895371d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.3234318494796753]], [[0.10518792271614075]], [[0.29617762565612793]], [[0.2048494517803192]], [[0.140651136636734]], [[0.2106209695339203]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.7354300022125244]], [[0.5847243070602417]], [[0.7040197849273682]], [[0.5520831942558289]], [[0.7446761727333069]], [[0.70814448595047]]], dtype='float32').reshape([6, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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


class PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a929203e3767fdbf9f28631d4035808b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_7837eb9c20d5f70502fabaaf04eced81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_0dd79751e325816d3c3f5495758327f0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 12096, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_968fbf05dda988a8cd65895aa4dbfa47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0dd79751e325816d3c3f5495758327f0
    def get_inputs(self):
        return [
            paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_af7abc8f3c07a92290a9b6159b7d361b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_291a3baba83aebcfbb0959b81157d6f7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8529a1be70d7f4c80d35871aebc24121(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_291a3baba83aebcfbb0959b81157d6f7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.4200737476348877, 0.2946609854698181]], [[0.47897306084632874, 0.29871895909309387]], [[0.34867048263549805, 0.2746659815311432]], [[0.022391103208065033, 0.1263965517282486]], [[0.00977354682981968, 0.026455778628587723]], [[0.39047884941101074, 0.10009819269180298]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.48737311363220215, 0.21872499585151672]], [[0.3569588363170624, 0.2996145188808441]], [[0.18173567950725555, 0.3620101511478424]], [[0.18104912340641022, 0.16540417075157166]], [[0.33887559175491333, 0.28572165966033936]], [[0.4271915853023529, 0.10836858302354813]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f7ca0670ac8140f9abb4a9bbf7b9ada2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_291a3baba83aebcfbb0959b81157d6f7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.08230464160442352, 0.17765505611896515]], [[0.41948506236076355, 0.2993577718734741]], [[0.26732832193374634, 0.14748147130012512]], [[0.4790646433830261, 0.15387652814388275]], [[0.47866836190223694, 0.19755837321281433]], [[0.2591554522514343, 0.37071239948272705]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.48737311363220215, 0.21872499585151672]], [[0.3569588363170624, 0.2996145188808441]], [[0.18173567950725555, 0.3620101511478424]], [[0.18104912340641022, 0.16540417075157166]], [[0.33887559175491333, 0.28572165966033936]], [[0.4271915853023529, 0.10836858302354813]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_c5c8a13052fe23a22c84469701570109(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 1, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_25cf1a551db28189c43b54388f12f005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5c8a13052fe23a22c84469701570109
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 21824, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.22964154183864594, 0.24316290020942688]], [[0.3824300467967987, 0.4855898916721344]], [[0.09435852617025375, 0.4004281461238861]], [[0.01536477915942669, 0.1013849675655365]], [[0.02149014361202717, 0.3536393940448761]], [[0.28038325905799866, 0.15163615345954895]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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


class PrimitiveOp_99bddb5fd63ecd93cc59d50704818428(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0b543be2087ce063597a56b11ea6ac5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([16]),
            paddle.to_tensor([0.021799949929118156, 0.0553545206785202, 0.1331569254398346, 0.12268710136413574, 0.39009812474250793, 0.20741882920265198, 0.16273048520088196, 0.31630992889404297, 0.1374664157629013, 0.005575885064899921, 0.18881046772003174, 0.18218941986560822, 0.18066896498203278, 0.39684590697288513, 0.053641725331544876, 0.24692824482917786], dtype='float32').reshape([16]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e38e3fde4ad5ba06b82555dd295fc739(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.021799949929118156, 0.0553545206785202, 0.1331569254398346, 0.12268710136413574, 0.39009812474250793, 0.20741882920265198, 0.16273048520088196, 0.31630992889404297, 0.1374664157629013, 0.005575885064899921, 0.18881046772003174, 0.18218941986560822, 0.18066896498203278, 0.39684590697288513, 0.053641725331544876, 0.24692824482917786], dtype='float32').reshape([16]),
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([16]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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


class PrimitiveOp_1d41c4f35c9bcc31e08344f76597024f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300], dtype='float32'),
            paddle.static.InputSpec(shape=[300], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e3862a2257d0a7b8b364fc904b57cdcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d41c4f35c9bcc31e08344f76597024f
    def get_inputs(self):
        return [
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e3862a2257d0a7b8b364fc904b57cdcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d41c4f35c9bcc31e08344f76597024f
    def get_inputs(self):
        return [
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_6e02c6ca3972717e5b5b70e1ebc41e36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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


class PrimitiveOp_ce10a710f7565f96c00a1389a91699d4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f05055f84295f40d92d0dcebbcf8a81a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce10a710f7565f96c00a1389a91699d4
    def get_inputs(self):
        return [
            paddle.uniform([1778, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1778, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4e357f0b3df6f5fe61b8d84f63ac48ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_4e357f0b3df6f5fe61b8d84f63ac48ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_4e357f0b3df6f5fe61b8d84f63ac48ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_4e357f0b3df6f5fe61b8d84f63ac48ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_4e357f0b3df6f5fe61b8d84f63ac48ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_4e357f0b3df6f5fe61b8d84f63ac48ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_4e357f0b3df6f5fe61b8d84f63ac48ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_4e357f0b3df6f5fe61b8d84f63ac48ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_4e357f0b3df6f5fe61b8d84f63ac48ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_4e357f0b3df6f5fe61b8d84f63ac48ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_4e357f0b3df6f5fe61b8d84f63ac48ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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


class PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_52d6c0904952e02947e994d334c88237(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b
    def get_inputs(self):
        return [
            paddle.uniform([3549, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_38dc692e168c69c6d9272c552d7727e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([3549, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f05055f84295f40d92d0dcebbcf8a81a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce10a710f7565f96c00a1389a91699d4
    def get_inputs(self):
        return [
            paddle.uniform([1778, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1778, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eecae8933709bf2506de121193ed0b13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4083593487739563, 0.0464426651597023, 0.19350320100784302, 0.0735485628247261], [0.04275652766227722, 0.42302489280700684, 0.1234317496418953, 0.4487559199333191], [0.3989906907081604, 0.423295795917511, 0.035104915499687195, 0.2857874929904938], [0.44080936908721924, 0.3033484220504761, 0.12672246992588043, 0.06597451865673065], [0.2818484306335449, 0.34505268931388855, 0.3520521819591522, 0.14433987438678741]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([[0.2586134672164917, 0.15547722578048706, 0.1751316338777542, 0.03089195117354393], [0.07368368655443192, 0.2442678064107895, 0.004473511129617691, 0.24331246316432953], [0.23028400540351868, 0.4056583642959595, 0.3727683126926422, 0.28531497716903687], [0.2434556633234024, 0.2522284686565399, 0.04619003087282181, 0.0900825560092926], [0.21415281295776367, 0.2867811918258667, 0.02609613910317421, 0.35222798585891724]], dtype='float32').reshape([5, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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


class PrimitiveOp_aedece1b298498e3693fa2b7e488785d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 5376, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_39ff3ebe61c4806058d6c962f4ef9475(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aedece1b298498e3693fa2b7e488785d
    def get_inputs(self):
        return [
            paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_536ed0de413ec3afb4ee079acceb8755(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.11813344806432724, 0.3100629150867462, 0.1310870349407196, 0.048055391758680344], [0.26968440413475037, 0.038270726799964905, 0.2786773145198822, 0.06464090198278427], [0.48029011487960815, 0.10376059263944626, 0.34819114208221436, 0.10710859298706055], [0.26968440413475037, 0.038270726799964905, 0.2786773145198822, 0.06464090198278427], [0.48029011487960815, 0.10376059263944626, 0.34819114208221436, 0.10710859298706055]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([[0.10199055820703506, 0.25440630316734314, 0.3121725618839264, 0.12423137575387955], [0.03548659756779671, 0.14141957461833954, 0.2776472568511963, 0.0903051495552063], [0.20749840140342712, 0.26365041732788086, 0.04141669720411301, 0.44134292006492615], [0.03548659756779671, 0.14141957461833954, 0.2776472568511963, 0.0903051495552063], [0.20749840140342712, 0.26365041732788086, 0.04141669720411301, 0.44134292006492615]], dtype='float32').reshape([5, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_b4e9b28607eff013e91bf04e8b23e889(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.10817418247461319], [0.21830850839614868], [0.1470475047826767], [0.2576325237751007], [0.06852712482213974], [0.39306512475013733], [0.006903389468789101], [0.004259239882230759], [0.36045563220977783]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.3870587646961212], [0.0602361299097538], [0.3039933145046234], [0.4867640435695648], [0.405436635017395], [0.3102510869503021], [0.4106326401233673], [0.37027329206466675], [0.4077361524105072]], dtype='float32').reshape([9, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e32c3fe8cf74a7896c71a84ee6048163(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.013448690064251423], [0.1428917944431305], [0.18482916057109833], [0.2410021275281906], [0.11816191673278809], [0.2892489433288574], [0.022538647055625916], [0.42789679765701294], [0.11527103185653687]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.3498869836330414], [0.26301130652427673], [0.3965403735637665], [0.37756359577178955], [0.17188876867294312], [0.06343194842338562], [0.4906500577926636], [0.49394193291664124], [0.43974825739860535]], dtype='float32').reshape([9, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c80fa5ee0ab2b73d17a44c009b818b00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.10817418247461319], [0.21830850839614868], [0.1470475047826767], [0.2576325237751007], [0.06852712482213974], [0.39306512475013733], [0.006903389468789101], [0.1592971235513687], [0.36045563220977783]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.16345298290252686], [0.03933706879615784], [0.26740995049476624], [0.15774424374103546], [0.22924834489822388], [0.3102510869503021], [0.4106326401233673], [0.31178075075149536], [0.4077361524105072]], dtype='float32').reshape([9, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6889ab982fee1064fb7a50e59c167b7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.013448690064251423], [0.39393821358680725], [0.41981005668640137], [0.2410021275281906], [0.28984448313713074], [0.2892489433288574], [0.022538647055625916], [0.48357874155044556], [0.11527103185653687]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.22238902747631073], [0.048635415732860565], [0.2709272503852844], [0.37756359577178955], [0.17188876867294312], [0.06343194842338562], [0.4906500577926636], [0.49394193291664124], [0.19016608595848083]], dtype='float32').reshape([9, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6dc561b3ed11d1f3c93fdc09d549e00a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3560943007469177], [0.2843972146511078], [0.19924697279930115], [0.39769965410232544], [0.1095288023352623], [0.4948180615901947], [0.43056219816207886], [0.004259239882230759], [0.4542475640773773]], dtype='float32').reshape([9, 1]),
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
class TestPrimitiveOp_32f379606fe67497f0bc134f5b82a199(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.09306798875331879], [0.1428917944431305], [0.18482916057109833], [0.4609587788581848], [0.11816191673278809], [0.3480241000652313], [0.3102295994758606], [0.42789679765701294], [0.15572763979434967]], dtype='float32').reshape([9, 1]),
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
class TestPrimitiveOp_dc4b3ba0bb9b026c01f18d05a51dfaf6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.01950223371386528], [0.03487321734428406], [0.0042560771107673645], [-0.029554974287748337], [-0.04360903054475784], [0.10207147896289825], [0.17654693126678467], [0.012095225043594837], [-0.09465660154819489]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.018700817599892616], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4675f232b75aa7c42c135c5f5ad91506(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3560943007469177], [0.2843972146511078], [0.19924697279930115], [0.39769965410232544], [0.1095288023352623], [0.4948180615901947], [0.43056219816207886], [0.1592971235513687], [0.4542475640773773]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.16345298290252686], [0.03933706879615784], [0.26740995049476624], [0.15774424374103546], [0.22924834489822388], [0.246979221701622], [0.08601068705320358], [0.31178075075149536], [0.10850615799427032]], dtype='float32').reshape([9, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d9cc7f9c0e5e0d12a06b2dac07ee6408(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.09306798875331879], [0.39393821358680725], [0.41981005668640137], [0.4609587788581848], [0.28984448313713074], [0.3480241000652313], [0.3102295994758606], [0.48357874155044556], [0.15572763979434967]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.22238902747631073], [0.048635415732860565], [0.2709272503852844], [0.2822781205177307], [0.03485541045665741], [0.011633493937551975], [0.3463441729545593], [0.45662522315979004], [0.19016608595848083]], dtype='float32').reshape([9, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_585781309cf679830c4742d8a8168ae6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.024912575259804726], [0.08461995422840118], [-0.01014829520136118], [0.042875390499830246], [-0.030527176335453987], [0.08337066322565079], [-0.012443330138921738], [-0.004109970293939114], [-0.011906796135008335]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.01950223371386528], [0.03487321734428406], [0.0042560771107673645], [-0.029554974287748337], [-0.04360903054475784], [0.08337066322565079], [0.17654693126678467], [0.012095225043594837], [-0.09465660154819489]], dtype='float32').reshape([9, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3ab043f9db6ded50b730a864a28ff48f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [-0.0], [-0.0], [0.22430932521820068], [0.0], [0.0], [-0.0]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[1.7828270196914673], [0.587884247303009], [1.4193884134292603], [1.689322590827942], [-0.4285314083099365], [0.0], [15.188077926635742], [3.9428985118865967], [-6.949796199798584]], dtype='float32').reshape([9, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1b2649627c56cc22f68d3cbea932371b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19d0a8a77ea8ac0dd0fb5dcf3895371d
    def get_inputs(self):
        return [
            paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_240608e199211c86bcbc0afed5a46001(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19d0a8a77ea8ac0dd0fb5dcf3895371d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.493002325296402]], [[0.021986098960042]], [[0.009986251592636108]], [[0.47580042481422424]], [[0.050395287573337555]], [[0.026547249406576157]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.5732123851776123]], [[0.7980857491493225]], [[0.6817842125892639]], [[0.7067649364471436]], [[0.592467725276947]], [[0.786468505859375]]], dtype='float32').reshape([6, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2bc7df90204aca0f4e5649e822f2f64f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19d0a8a77ea8ac0dd0fb5dcf3895371d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.38881805539131165]], [[0.16338205337524414]], [[0.16764523088932037]], [[0.06548044085502625]], [[0.20316286385059357]], [[0.42358702421188354]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.6939191818237305]], [[0.6774009466171265]], [[0.5296899080276489]], [[0.6363528966903687]], [[0.6614072918891907]], [[0.5798671245574951]]], dtype='float32').reshape([6, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_ef457a8c2827717ad98c98ab5f8f38cd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_14bae7859044d7ac7a1e855f2042d793(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef457a8c2827717ad98c98ab5f8f38cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_054ef6f8e993a29d71a4ae95e1ef63f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce10a710f7565f96c00a1389a91699d4
    def get_inputs(self):
        return [
            paddle.uniform([5640, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([5640, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ff6b78dacefc2897aa15247e539448e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_ff6b78dacefc2897aa15247e539448e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_ff6b78dacefc2897aa15247e539448e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_ff6b78dacefc2897aa15247e539448e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_ff6b78dacefc2897aa15247e539448e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_ff6b78dacefc2897aa15247e539448e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_ff6b78dacefc2897aa15247e539448e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_ff6b78dacefc2897aa15247e539448e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_ff6b78dacefc2897aa15247e539448e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_ff6b78dacefc2897aa15247e539448e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_ff6b78dacefc2897aa15247e539448e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_03b1906bab8cf1f48b4606ade5191acd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b
    def get_inputs(self):
        return [
            paddle.uniform([11109, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4f0222eca32e6403f4049a2fb3b688ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([11109, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_054ef6f8e993a29d71a4ae95e1ef63f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce10a710f7565f96c00a1389a91699d4
    def get_inputs(self):
        return [
            paddle.uniform([5640, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([5640, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b8813d1ff55826a67087c1015e0825b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.40535181760787964, 0.18487544357776642, 0.12082479894161224, 0.15880312025547028], [0.20091252028942108, 0.08696173131465912, 0.253089040517807, 0.31328579783439636], [0.25166910886764526, 0.34895312786102295, 0.10745351016521454, 0.08950001746416092], [0.20091252028942108, 0.08696173131465912, 0.253089040517807, 0.31328579783439636], [0.25166910886764526, 0.34895312786102295, 0.10745351016521454, 0.08950001746416092], [0.11380904167890549, 0.49813464283943176, 0.28097593784332275, 0.20141464471817017], [0.11380904167890549, 0.49813464283943176, 0.28097593784332275, 0.20141464471817017]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([[0.06191736087203026, 0.3354967534542084, 0.11539153009653091, 0.05804868042469025], [0.3243649899959564, 0.2906089127063751, 0.09762857109308243, 0.3227904438972473], [0.48022186756134033, 0.03066052496433258, 0.0535271018743515, 0.20987284183502197], [0.3243649899959564, 0.2906089127063751, 0.09762857109308243, 0.3227904438972473], [0.48022186756134033, 0.03066052496433258, 0.0535271018743515, 0.20987284183502197], [0.09828855842351913, 0.20395898818969727, 0.08289018273353577, 0.36240696907043457], [0.09828855842351913, 0.20395898818969727, 0.08289018273353577, 0.36240696907043457]], dtype='float32').reshape([7, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0e93be1b02e7657a17ee401eb2ea5cf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0e93be1b02e7657a17ee401eb2ea5cf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6a0d88c3b31ac0ec78be6d8038ebdb82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_18b385f2773e20549fa68a071089e091(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a61d16187b170698f9108434bcba5729(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.21110419929027557, 0.35089540481567383, 0.2971991300582886, 0.16651640832424164, 0.40372434258461, 0.39073458313941956], dtype='float32').reshape([6]),
            paddle.to_tensor([0.4102131128311157, 0.047395769506692886, 0.13399538397789001, 0.29094526171684265, 0.009467806667089462, 0.18842779099941254], dtype='float32').reshape([6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_edbc75489aa0d60ff9ffddf81b5d417f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.038237329572439194, 0.16254805028438568, 0.004502696916460991, 0.1902225911617279, 0.19850163161754608, 0.19214455783367157], dtype='float32').reshape([6]),
            paddle.to_tensor([0.1450636386871338, 0.25107648968696594, 0.23796823620796204, 0.386831134557724, 0.3538130819797516, 0.40666136145591736], dtype='float32').reshape([6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7b795e41a4c1fbfd6bdca0dabd29e280(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3418285846710205, 0.42906653881073, 0.04830992594361305, 0.33282768726348877, 0.28770944476127625, 0.12256492674350739], dtype='float32').reshape([6]),
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
class TestPrimitiveOp_29b20e447022bdff8e44bd46f99956f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4553065598011017, 0.16179899871349335, 0.38409265875816345, 0.37823209166526794, 0.21884433925151825, 0.2555798888206482], dtype='float32').reshape([6]),
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
class TestPrimitiveOp_924d332831e3e6d2743abd4f612bfae5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3418285846710205, 0.35089540481567383, 0.04830992594361305, 0.29094526171684265, 0.28770944476127625, 0.12256492674350739], dtype='float32').reshape([6]),
            paddle.to_tensor([0.4102131128311157, 0.20509734749794006, 0.16082343459129333, 0.4748803675174713, 0.07652277499437332, 0.20391766726970673], dtype='float32').reshape([6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f65bc9f9e3b8dce8341ff33f559d2788(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1450636386871338, 0.16179899871349335, 0.23796823620796204, 0.37823209166526794, 0.21884433925151825, 0.2555798888206482], dtype='float32').reshape([6]),
            paddle.to_tensor([0.4793972969055176, 0.25107648968696594, 0.23796823620796204, 0.386831134557724, 0.3538130819797516, 0.48301398754119873], dtype='float32').reshape([6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_429ff7dce7d8b733136f0037a2e1b008(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4102131128311157, 0.35089540481567383, 0.2971991300582886, 0.29094526171684265, 0.40372434258461, 0.39073458313941956], dtype='float32').reshape([6]),
            paddle.to_tensor([0.4102131128311157, 0.047395769506692886, 0.13399538397789001, 0.29094526171684265, 0.009467806667089462, 0.18842779099941254], dtype='float32').reshape([6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7b9fa4029744a810baeb3f5f0ca16f31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1450636386871338, 0.25107648968696594, 0.23796823620796204, 0.386831134557724, 0.3538130819797516, 0.40666136145591736], dtype='float32').reshape([6]),
            paddle.to_tensor([0.1450636386871338, 0.25107648968696594, 0.23796823620796204, 0.386831134557724, 0.3538130819797516, 0.40666136145591736], dtype='float32').reshape([6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_570b6914e2aba624653da6012b0796ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.0001800392201403156, 0.0017102111596614122, -0.033292002975940704, -0.01555620227009058, 0.04252384603023529, 0.018502388149499893], dtype='float32').reshape([6]),
            paddle.to_tensor([0.0, -0.0, -0.0, 0.0, -0.0, 0.0], dtype='float32').reshape([6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_27aab46b52317ccf07492f1442f7946f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.31065866351127625, 0.1991455852985382, 0.2155972570180893, 0.22873082756996155, 0.20659607648849487, 0.28958117961883545], dtype='float32').reshape([6]),
            paddle.to_tensor([0.3380919098854065, 0.31708192825317383, 0.10456667840480804, 0.40385401248931885, 0.18211610615253448, 0.16324129700660706], dtype='float32').reshape([6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a090f43ee249e6724a0d17519a81fe5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.09165048599243164, 0.20681226253509521, 0.12123546749353409, 0.28852686285972595, 0.27615734934806824, 0.29940295219421387], dtype='float32').reshape([6]),
            paddle.to_tensor([0.46735191345214844, 0.1579810380935669, 0.2361459732055664, 0.32347702980041504, 0.1181660071015358, 0.36929693818092346], dtype='float32').reshape([6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ed0a81c75952d9adc75ed473b849e8fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4102131128311157, 0.42906653881073, 0.2971991300582886, 0.33282768726348877, 0.40372434258461, 0.39073458313941956], dtype='float32').reshape([6]),
            paddle.to_tensor([0.3343552052974701, 0.047395769506692886, 0.13399538397789001, 0.29094526171684265, 0.009467806667089462, 0.18842779099941254], dtype='float32').reshape([6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_94a4953afe90942d79d977cf7910b97b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4553065598011017, 0.25107648968696594, 0.38409265875816345, 0.386831134557724, 0.3538130819797516, 0.40666136145591736], dtype='float32').reshape([6]),
            paddle.to_tensor([0.1450636386871338, 0.15416307747364044, 0.08819928020238876, 0.2687219977378845, 0.017487674951553345, 0.40666136145591736], dtype='float32').reshape([6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6ae434afba7a1bab2098e08b8ab64305(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.3008045256137848, 1.5367159843444824, -0.36336562037467957, -0.9140454530715942, 0.8092215657234192, 0.3435162603855133], dtype='float32').reshape([6]),
            paddle.to_tensor([1.078359842300415, -1.286978840827942, -0.6100871562957764, 0.5642430186271667, -1.1955293416976929, -0.7561135292053223], dtype='float32').reshape([6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4a001f42da04571169401ec446fcd412(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce10a710f7565f96c00a1389a91699d4
    def get_inputs(self):
        return [
            paddle.uniform([1797, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1797, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2b8b676d4170845c778ee2d01e793bd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_2b8b676d4170845c778ee2d01e793bd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_2b8b676d4170845c778ee2d01e793bd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_2b8b676d4170845c778ee2d01e793bd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_2b8b676d4170845c778ee2d01e793bd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_2b8b676d4170845c778ee2d01e793bd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_2b8b676d4170845c778ee2d01e793bd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_2b8b676d4170845c778ee2d01e793bd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_2b8b676d4170845c778ee2d01e793bd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_2b8b676d4170845c778ee2d01e793bd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_2b8b676d4170845c778ee2d01e793bd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_52d6c0904952e02947e994d334c88237(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b
    def get_inputs(self):
        return [
            paddle.uniform([3549, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_38dc692e168c69c6d9272c552d7727e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([3549, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4a001f42da04571169401ec446fcd412(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce10a710f7565f96c00a1389a91699d4
    def get_inputs(self):
        return [
            paddle.uniform([1797, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1797, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_9fec7ada1680728cff3c134be87b6aa2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8400, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0bd7d880fc7306d155fe38898579c30a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9fec7ada1680728cff3c134be87b6aa2
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_958bd85215449bcd203b84066797db5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([24]),
            paddle.to_tensor([0.14985322952270508, 0.3614925444126129, 0.39877936244010925, 0.053330495953559875, 0.13969476521015167, 0.0027347251307219267, 0.28718096017837524, 0.06502088159322739, 0.490936279296875, 0.3843059539794922, 0.1196126788854599, 0.15985898673534393, 0.024629564955830574, 0.3084601163864136, 0.15704958140850067, 0.3779698312282562, 0.4783381223678589, 0.23180651664733887, 0.2301483005285263, 0.21951468288898468, 0.4981519877910614, 0.11176123470067978, 0.3388000726699829, 0.1751078963279724], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_359d7a62bab0a45964a45ed24d1f585c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.14985322952270508, 0.3614925444126129, 0.39877936244010925, 0.053330495953559875, 0.13969476521015167, 0.0027347251307219267, 0.28718096017837524, 0.06502088159322739, 0.490936279296875, 0.3843059539794922, 0.1196126788854599, 0.15985898673534393, 0.024629564955830574, 0.3084601163864136, 0.15704958140850067, 0.3779698312282562, 0.4783381223678589, 0.23180651664733887, 0.2301483005285263, 0.21951468288898468, 0.4981519877910614, 0.11176123470067978, 0.3388000726699829, 0.1751078963279724], dtype='float32').reshape([24]),
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_9d26f66172fd09bff7d691395314291a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce10a710f7565f96c00a1389a91699d4
    def get_inputs(self):
        return [
            paddle.uniform([1501, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1501, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ab7d5a8762dacbe875f12504bd67362f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_ab7d5a8762dacbe875f12504bd67362f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_ab7d5a8762dacbe875f12504bd67362f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_ab7d5a8762dacbe875f12504bd67362f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_ab7d5a8762dacbe875f12504bd67362f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_ab7d5a8762dacbe875f12504bd67362f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_ab7d5a8762dacbe875f12504bd67362f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_ab7d5a8762dacbe875f12504bd67362f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_ab7d5a8762dacbe875f12504bd67362f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_ab7d5a8762dacbe875f12504bd67362f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_ab7d5a8762dacbe875f12504bd67362f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_3ed75efcd84ec090d0b5f8a4bdcffa95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b
    def get_inputs(self):
        return [
            paddle.uniform([3024, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a1d0e3e7e00154ab42eff6103c79fdb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([3024, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9d26f66172fd09bff7d691395314291a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce10a710f7565f96c00a1389a91699d4
    def get_inputs(self):
        return [
            paddle.uniform([1501, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1501, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_95329544739f16b1ba6257c58390f2ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([4]),
            paddle.to_tensor([0.31062570214271545, 0.34576329588890076, 0.1544123739004135, 0.4026666283607483], dtype='float32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0344c476ae89de1a5be9f95cb2f83303(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.31062570214271545, 0.34576329588890076, 0.1544123739004135, 0.4026666283607483], dtype='float32').reshape([4]),
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_1917ea0a108920c4069ffa6dfbed0f79(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8cdd6eeef482a5f5c2da614c8ee3b44b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1917ea0a108920c4069ffa6dfbed0f79
    def get_inputs(self):
        return [
            paddle.to_tensor([4], dtype='int32').reshape([1]),
            paddle.to_tensor([2], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_56c4a2e44901467a3e8ec6b3b0b1c5e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1917ea0a108920c4069ffa6dfbed0f79
    def get_inputs(self):
        return [
            paddle.to_tensor([7], dtype='int32').reshape([1]),
            paddle.to_tensor([3], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_f38049eed6836134ebfa696e207c5ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.10230942815542221, 0.4656737744808197, 0.23023895919322968, 0.3134593963623047], [0.037858448922634125, 0.4459499418735504, 0.04106490686535835, 0.34768176078796387], [0.3960135579109192, 0.30466198921203613, 0.1388903111219406, 0.2818836569786072], [0.43273648619651794, 0.2460307776927948, 0.4382016062736511, 0.267194002866745], [0.43273648619651794, 0.2460307776927948, 0.4382016062736511, 0.267194002866745], [0.3960135579109192, 0.30466198921203613, 0.1388903111219406, 0.2818836569786072]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([[0.036272622644901276, 0.019473768770694733, 0.2123793065547943, 0.2291458547115326], [0.31556394696235657, 0.4416036903858185, 0.21069373190402985, 0.10256848484277725], [0.19106312096118927, 0.24031859636306763, 0.019208991900086403, 0.38589227199554443], [0.49473837018013, 0.3024352490901947, 0.3373301327228546, 0.3158513307571411], [0.49473837018013, 0.3024352490901947, 0.3373301327228546, 0.3158513307571411], [0.19106312096118927, 0.24031859636306763, 0.019208991900086403, 0.38589227199554443]], dtype='float32').reshape([6, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0226ae1e09d1d8c97bd533dc8f9c09a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2260180413722992, 0.19949793815612793, 0.21764975786209106, 0.3027788996696472], [0.14532725512981415, 0.257521390914917, 0.46131858229637146, 0.07517838478088379], [0.1859636753797531, 0.011614393442869186, 0.09635768085718155, 0.33755823969841003], [0.017686065286397934, 0.058734674006700516, 0.36337247490882874, 0.492013156414032], [0.2260180413722992, 0.19949793815612793, 0.21764975786209106, 0.3027788996696472]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([[0.3445029556751251, 0.45443597435951233, 0.2382742315530777, 0.03435212001204491], [0.10079831629991531, 0.2235482782125473, 0.2061411738395691, 0.343816339969635], [0.35300323367118835, 0.1610371619462967, 0.1562039703130722, 0.1116558089852333], [0.19510143995285034, 0.2744866907596588, 0.15754464268684387, 0.30516183376312256], [0.3445029556751251, 0.45443597435951233, 0.2382742315530777, 0.03435212001204491]], dtype='float32').reshape([5, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_56bd3d7f1023b32564a39a9793b4d622(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_55830e6682332d190b9d1c2ce9af6c22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.16136038303375244]], dtype='float32').reshape([1, 1]),
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
class TestPrimitiveOp_2f6dd69d4ca2c39b86352cb3f1e583f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.43908387422561646]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.235838383436203]], dtype='float32').reshape([1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7610e93fa1ad6e4bfd84bcd7f5be0c05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.26837071776390076]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.08844736218452454]], dtype='float32').reshape([1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2f6dd69d4ca2c39b86352cb3f1e583f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.43908387422561646]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.235838383436203]], dtype='float32').reshape([1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_55830e6682332d190b9d1c2ce9af6c22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.16136038303375244]], dtype='float32').reshape([1, 1]),
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
class TestPrimitiveOp_e9881d22393f4a735ae67492585cfb50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4695817530155182]], dtype='float32').reshape([1, 1]),
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
class TestPrimitiveOp_3e48a0c956cc032672e5bd9727a2e8b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.04066997021436691]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7610e93fa1ad6e4bfd84bcd7f5be0c05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.26837071776390076]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.08844736218452454]], dtype='float32').reshape([1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e9881d22393f4a735ae67492585cfb50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4695817530155182]], dtype='float32').reshape([1, 1]),
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
class TestPrimitiveOp_67fc21e1adfd0ed06b07f914805dafd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.04341842234134674]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.04066997021436691]], dtype='float32').reshape([1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1a9003ea910d2a73a12256a9f2d8bb55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[1.9366984367370605]], dtype='float32').reshape([1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_db2a0e654bd120967d8c03b1c7def925(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.35055011510849], [0.09527037292718887], [0.32676854729652405], [0.12946005165576935], [0.37342819571495056], [0.03219883143901825]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.3338686227798462], [0.07760456204414368], [0.3587840497493744], [0.40559735894203186], [0.3328937888145447], [0.46134132146835327]], dtype='float32').reshape([6, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2a2645d16c91edb48b527d9c93281b3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.405924528837204], [0.21171081066131592], [0.1269102543592453], [0.3708653151988983], [0.14785419404506683], [0.3854176700115204]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.39591267704963684], [0.25403302907943726], [0.3512172996997833], [0.486686110496521], [0.46994784474372864], [0.44575998187065125]], dtype='float32').reshape([6, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_915e5387cd386f549cba4fec6cc8be82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.35055011510849], [0.09527037292718887], [0.40782997012138367], [0.12946005165576935], [0.37342819571495056], [0.3238355815410614]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.3338686227798462], [0.0581473633646965], [0.27728891372680664], [0.40559735894203186], [0.2994241714477539], [0.46134132146835327]], dtype='float32').reshape([6, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ef4aad64752b01997bb2010e321de425(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.405924528837204], [0.31001538038253784], [0.2668556869029999], [0.47325971722602844], [0.4950177073478699], [0.3854176700115204]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.39591267704963684], [0.11558108776807785], [0.3512172996997833], [0.486686110496521], [0.46994784474372864], [0.3137425482273102]], dtype='float32').reshape([6, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6ed50df2234a689874f558a0eeb37521(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.355660617351532], [0.35937029123306274], [0.32676854729652405], [0.3721979558467865], [0.4757102429866791], [0.03219883143901825]], dtype='float32').reshape([6, 1]),
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
class TestPrimitiveOp_b38b84a5259e83db24292e12958eaf6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.43543845415115356], [0.21171081066131592], [0.1269102543592453], [0.3708653151988983], [0.14785419404506683], [0.4037357568740845]], dtype='float32').reshape([6, 1]),
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
class TestPrimitiveOp_25d4f96fc7acde48b7c5d2971da27b74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.09359443187713623], [-0.004706964362412691], [-0.01218957919627428], [-0.004550507292151451], [-0.022184086963534355], [0.0035779764875769615]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.00016701262211427093], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_be31ea024f01e1a84533aad0db482915(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.355660617351532], [0.35937029123306274], [0.40782997012138367], [0.3721979558467865], [0.4757102429866791], [0.3238355815410614]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.06195531785488129], [0.0581473633646965], [0.27728891372680664], [0.39517515897750854], [0.2994241714477539], [0.35186487436294556]], dtype='float32').reshape([6, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c516f20c732ca49b97c315878be9b0ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.43543845415115356], [0.31001538038253784], [0.2668556869029999], [0.47325971722602844], [0.4950177073478699], [0.4037357568740845]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.11733926832675934], [0.11558108776807785], [0.09014913439750671], [0.011464092880487442], [0.316177636384964], [0.3137425482273102]], dtype='float32').reshape([6, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f2fe03fd2101fb0041be836e9ec1cf2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.09342741966247559], [0.058568064123392105], [0.02306745946407318], [-0.010610772296786308], [0.03152701258659363], [-0.002522445982322097]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.09342741966247559], [-0.004706964362412691], [-0.01218957919627428], [-0.004550507292151451], [-0.022184086963534355], [0.0035779764875769615]], dtype='float32').reshape([6, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_771057f4b138ec75b770be8019c2538a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0017876189667731524], [-0.0], [-0.0], [-0.0], [-0.0], [0.0]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.0], [1.0803674459457397], [1.52843177318573], [0.5711426734924316], [1.703653335571289], [2.418455123901367]], dtype='float32').reshape([6, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c8cd9d9c29e747e89cc21bc77f32ee53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4666564464569092, 0.3648544251918793, 0.18811792135238647, 0.14878439903259277], [0.4952784776687622, 0.14019908010959625, 0.14755922555923462, 0.26648133993148804], [0.45726096630096436, 0.4442957639694214, 0.05174024775624275, 0.17932143807411194], [0.355541855096817, 0.4766886830329895, 0.16455477476119995, 0.24976997077465057]], dtype='float32').reshape([4, 4]),
            paddle.to_tensor([[0.2993108332157135, 0.2873929440975189, 0.3280841112136841, 0.0652041807770729], [0.004542017355561256, 0.09385919570922852, 0.0289857666939497, 0.4194986820220947], [0.3477599620819092, 0.3219774067401886, 0.10543779283761978, 0.32350409030914307], [0.026139285415410995, 0.1471622884273529, 0.12444959580898285, 0.19544094800949097]], dtype='float32').reshape([4, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_ad01ba209aa563c77d8a0f5ecf6c7ca2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f3a8ea46e19f84a1da5b00db7a84c3ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce10a710f7565f96c00a1389a91699d4
    def get_inputs(self):
        return [
            paddle.uniform([2078, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2078, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_05fe2a05965b6dd6ff64bff5b5223132(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_05fe2a05965b6dd6ff64bff5b5223132(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_05fe2a05965b6dd6ff64bff5b5223132(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_05fe2a05965b6dd6ff64bff5b5223132(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_05fe2a05965b6dd6ff64bff5b5223132(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_05fe2a05965b6dd6ff64bff5b5223132(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_05fe2a05965b6dd6ff64bff5b5223132(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_05fe2a05965b6dd6ff64bff5b5223132(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_05fe2a05965b6dd6ff64bff5b5223132(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_05fe2a05965b6dd6ff64bff5b5223132(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_05fe2a05965b6dd6ff64bff5b5223132(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_15aa17eea687877cf988c12a2c059c6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b
    def get_inputs(self):
        return [
            paddle.uniform([4116, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_358bc8c1eb98b6e964028dcc0f56dec2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([4116, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f3a8ea46e19f84a1da5b00db7a84c3ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce10a710f7565f96c00a1389a91699d4
    def get_inputs(self):
        return [
            paddle.uniform([2078, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2078, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a39b70f293f0a1c4860e458c41b36f2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.05868971347808838, 0.0931183323264122, 0.4817720353603363, 0.2228946089744568], [0.05868971347808838, 0.0931183323264122, 0.4817720353603363, 0.2228946089744568], [0.1748245358467102, 0.4562341272830963, 0.3886791169643402, 0.12167710065841675], [0.04895717278122902, 0.004515435546636581, 0.1815401315689087, 0.42079052329063416], [0.4375079870223999, 0.014157120138406754, 0.24924929440021515, 0.234329953789711], [0.17357571423053741, 0.2881641089916229, 0.16192957758903503, 0.2890987992286682], [0.18074443936347961, 0.18766231834888458, 0.4655025601387024, 0.06498397141695023]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([[0.427163302898407, 0.26374343037605286, 0.46053290367126465, 0.00455693993717432], [0.427163302898407, 0.26374343037605286, 0.46053290367126465, 0.00455693993717432], [0.030603915452957153, 0.29073208570480347, 0.43393781781196594, 0.03389497473835945], [0.2495403289794922, 0.18423983454704285, 0.046639569103717804, 0.4862843453884125], [0.23021572828292847, 0.21562013030052185, 0.24692431092262268, 0.44247740507125854], [0.20590561628341675, 0.06558937579393387, 0.27467429637908936, 0.04392644390463829], [0.4704324007034302, 0.39693382382392883, 0.07851158082485199, 0.20433372259140015]], dtype='float32').reshape([7, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_40f637a1c203bb54d195c3251937c47a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_19bcde026d279fc2319de30a93e14494(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef457a8c2827717ad98c98ab5f8f38cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_df790fb541b5b61c884396a582da66b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce10a710f7565f96c00a1389a91699d4
    def get_inputs(self):
        return [
            paddle.uniform([4653, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4653, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9a95620d85ec0e60f4418538d2dbddc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_9a95620d85ec0e60f4418538d2dbddc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_9a95620d85ec0e60f4418538d2dbddc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_9a95620d85ec0e60f4418538d2dbddc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_9a95620d85ec0e60f4418538d2dbddc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_9a95620d85ec0e60f4418538d2dbddc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_9a95620d85ec0e60f4418538d2dbddc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_9a95620d85ec0e60f4418538d2dbddc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_9a95620d85ec0e60f4418538d2dbddc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_9a95620d85ec0e60f4418538d2dbddc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_9a95620d85ec0e60f4418538d2dbddc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_fcb8107f4be5eea7babae7f8cd89696e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b
    def get_inputs(self):
        return [
            paddle.uniform([9261, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2f56d38e40cb258115066d09088cf23a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([9261, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_df790fb541b5b61c884396a582da66b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce10a710f7565f96c00a1389a91699d4
    def get_inputs(self):
        return [
            paddle.uniform([4653, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4653, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_48979a4cc4e5b6777551aef446d39284(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce10a710f7565f96c00a1389a91699d4
    def get_inputs(self):
        return [
            paddle.uniform([1022, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1022, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d1c034f79e2df94b85abccc08ae85593(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_d1c034f79e2df94b85abccc08ae85593(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_d1c034f79e2df94b85abccc08ae85593(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_d1c034f79e2df94b85abccc08ae85593(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_d1c034f79e2df94b85abccc08ae85593(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_d1c034f79e2df94b85abccc08ae85593(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_d1c034f79e2df94b85abccc08ae85593(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_d1c034f79e2df94b85abccc08ae85593(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_d1c034f79e2df94b85abccc08ae85593(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_d1c034f79e2df94b85abccc08ae85593(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_d1c034f79e2df94b85abccc08ae85593(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_21a50b949a46359c33f1a60837b3d774(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b
    def get_inputs(self):
        return [
            paddle.uniform([2100, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_67dd1ecbca2884ee78c7cd1a7eea2215(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([2100, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_48979a4cc4e5b6777551aef446d39284(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce10a710f7565f96c00a1389a91699d4
    def get_inputs(self):
        return [
            paddle.uniform([1022, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1022, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_515f52247eb0925eb57e78c260d58f96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_291a3baba83aebcfbb0959b81157d6f7
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_faa77adc4925f43471195fed3e1a9874(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.030757920816540718, 0.30942824482917786, 0.09276753664016724, 0.44252586364746094], [0.4536651074886322, 0.4457365870475769, 0.14740948379039764, 0.4085237383842468], [0.4536651074886322, 0.4457365870475769, 0.14740948379039764, 0.4085237383842468], [0.29568400979042053, 0.020557958632707596, 0.229517862200737, 0.2792818248271942], [0.02849540486931801, 0.052373725920915604, 0.2668256163597107, 0.037516992539167404], [0.44800955057144165, 0.4503563940525055, 0.4647688865661621, 0.37141361832618713]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([[0.04741102084517479, 0.12851227819919586, 0.42099282145500183, 0.06301568448543549], [0.06640581786632538, 0.46593543887138367, 0.36126700043678284, 0.2646188735961914], [0.06640581786632538, 0.46593543887138367, 0.36126700043678284, 0.2646188735961914], [0.12631738185882568, 0.42651644349098206, 0.20479850471019745, 0.09247107803821564], [0.17669986188411713, 0.0070557976141572, 0.24092066287994385, 0.1861073076725006], [0.3691565692424774, 0.4530242085456848, 0.4404996931552887, 0.09307760745286942]], dtype='float32').reshape([6, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_293d93bbcf6e674049780843ff641a48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([100, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([100, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_32469e7c3a76900c5c34599e508cf549(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, 1, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ac27d31991e62955ca1970b03ba2bc45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32469e7c3a76900c5c34599e508cf549
    def get_inputs(self):
        return [
            paddle.uniform([100, 1, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1.3999216556549072, 0.4594396948814392, 3.3227317333221436, 1.2254887819290161], [0.4531983733177185, 2.966006278991699, 1.7263309955596924, 58.01933670043945]], dtype='float32').reshape([2, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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


class PrimitiveOp_8b891c3c0afc2d4183664ce45c3a7b49(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 6069, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ea82364bb78e0e4eedc177c19e5452e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b891c3c0afc2d4183664ce45c3a7b49
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5506a7d5add76bd5c93be0c3ec95da8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([300, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([300, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_af27d48e9942876178a44f8989dc4d03(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300, 1, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e1d3a518634db6e7dd57d54679b71335(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af27d48e9942876178a44f8989dc4d03
    def get_inputs(self):
        return [
            paddle.uniform([300, 1, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[2.259181022644043, 0.43287745118141174, 1.924061894416809, 1.5772923231124878], [0.07533172518014908, 0.9091368317604065, 0.9618554711341858, 3.694631576538086]], dtype='float32').reshape([2, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7a90f3a423a71068378b1eb112b05497(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.19564540684223175], [0.034317974001169205], [0.29465341567993164], [0.24096974730491638], [0.33738017082214355]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.20026913285255432], [0.41332828998565674], [0.18956781923770905], [0.2602648437023163], [0.48564788699150085]], dtype='float32').reshape([5, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d1da23cbe8ea5a5e3cb742276145e27b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.034618474543094635], [0.3562788665294647], [0.11915521323680878], [0.15779650211334229], [0.047968797385692596]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.22313657402992249], [0.2731638550758362], [0.2620335519313812], [0.48791664838790894], [0.16293737292289734]], dtype='float32').reshape([5, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a8ebb5f145608899474b6b6a53eda95d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.19564540684223175], [0.034317974001169205], [0.29465341567993164], [0.24096974730491638], [0.33738017082214355]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.20026913285255432], [0.18348328769207], [0.18956781923770905], [0.2602648437023163], [0.2216607630252838]], dtype='float32').reshape([5, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_899829f0ae9fe5562eda5a3ea29cc1aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.12423448264598846], [0.3562788665294647], [0.11915521323680878], [0.15779650211334229], [0.047968797385692596]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.22313657402992249], [0.19915108382701874], [0.2620335519313812], [0.48791664838790894], [0.14740948379039764]], dtype='float32').reshape([5, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_647b3b41e294fcdcad24d8c77ca09182(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.37746039032936096], [0.16417308151721954], [0.38406670093536377], [0.27725130319595337], [0.4067143201828003]], dtype='float32').reshape([5, 1]),
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
class TestPrimitiveOp_662916f4af1fc905fcffd3634f4f3b50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.034618474543094635], [0.3611237704753876], [0.4964972734451294], [0.23624840378761292], [0.08820351958274841]], dtype='float32').reshape([5, 1]),
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
class TestPrimitiveOp_693aaa42f72c24eb5e6e62025a443624(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.00875263661146164], [-0.045353688299655914], [0.07074305415153503], [0.018018260598182678], [-0.0056082080118358135]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8c7305a0518e93f0151c6c562fadf8d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.37746039032936096], [0.16417308151721954], [0.38406670093536377], [0.27725130319595337], [0.4067143201828003]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.07452657073736191], [0.18348328769207], [0.12519223988056183], [0.003253175877034664], [0.2216607630252838]], dtype='float32').reshape([5, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_65fa390c7579f2dbc4260208736e8715(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.12423448264598846], [0.3611237704753876], [0.4964972734451294], [0.23624840378761292], [0.08820351958274841]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.00723513076081872], [0.19915108382701874], [0.16522662341594696], [0.19373512268066406], [0.14740948379039764]], dtype='float32').reshape([5, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_16488c14b697880480ab4e587ac7764b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.03544306010007858], [-0.0031277260277420282], [0.08575750887393951], [0.01164855994284153], [-0.010956274345517159]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.00875263661146164], [-0.045353688299655914], [0.07074305415153503], [0.018018260598182678], [-0.0056082080118358135]], dtype='float32').reshape([5, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_51b38bb19e03d14f9fb4f619552ab72f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [-0.0], [0.0], [0.0], [-0.0]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.7530507445335388], [-13.500530242919922], [0.175080344080925], [-0.5468230247497559], [0.4881281852722168]], dtype='float32').reshape([5, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f4eed4ac0bd1910c2ef37732dde94533(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef457a8c2827717ad98c98ab5f8f38cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_b3721a6740381517ff2bb35005a88b57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce10a710f7565f96c00a1389a91699d4
    def get_inputs(self):
        return [
            paddle.uniform([2305, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2305, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fc4e6c7dcbb4dd8b4abd3c0a24747476(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_fc4e6c7dcbb4dd8b4abd3c0a24747476(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_fc4e6c7dcbb4dd8b4abd3c0a24747476(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_fc4e6c7dcbb4dd8b4abd3c0a24747476(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_fc4e6c7dcbb4dd8b4abd3c0a24747476(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_fc4e6c7dcbb4dd8b4abd3c0a24747476(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_fc4e6c7dcbb4dd8b4abd3c0a24747476(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_fc4e6c7dcbb4dd8b4abd3c0a24747476(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_fc4e6c7dcbb4dd8b4abd3c0a24747476(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_fc4e6c7dcbb4dd8b4abd3c0a24747476(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_fc4e6c7dcbb4dd8b4abd3c0a24747476(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_0cf1b79d6b0bda99fe30897e159b4ad8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b
    def get_inputs(self):
        return [
            paddle.uniform([4725, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_482a57e3f8a9d1754788fd87985c8f0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([4725, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b3721a6740381517ff2bb35005a88b57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce10a710f7565f96c00a1389a91699d4
    def get_inputs(self):
        return [
            paddle.uniform([2305, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2305, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9fd9b58bf7fd6573eb729b1d254aefd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce10a710f7565f96c00a1389a91699d4
    def get_inputs(self):
        return [
            paddle.uniform([3053, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3053, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_668471aea871302bc241cd12761c662f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_668471aea871302bc241cd12761c662f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_668471aea871302bc241cd12761c662f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_668471aea871302bc241cd12761c662f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_668471aea871302bc241cd12761c662f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_668471aea871302bc241cd12761c662f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_668471aea871302bc241cd12761c662f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_668471aea871302bc241cd12761c662f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_668471aea871302bc241cd12761c662f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_668471aea871302bc241cd12761c662f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_668471aea871302bc241cd12761c662f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_8d5f52ddf8383c305d82d62c38a7f453(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b
    def get_inputs(self):
        return [
            paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fc25a83b16b04b6ad5a3de78a8e76b5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9fd9b58bf7fd6573eb729b1d254aefd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce10a710f7565f96c00a1389a91699d4
    def get_inputs(self):
        return [
            paddle.uniform([3053, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3053, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_21d5152b9f03c359a771aa42b28c95e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce10a710f7565f96c00a1389a91699d4
    def get_inputs(self):
        return [
            paddle.uniform([3712, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3712, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1b56eac4fe5d134e3350a59139d67d6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_1b56eac4fe5d134e3350a59139d67d6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_1b56eac4fe5d134e3350a59139d67d6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_1b56eac4fe5d134e3350a59139d67d6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_1b56eac4fe5d134e3350a59139d67d6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_1b56eac4fe5d134e3350a59139d67d6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_1b56eac4fe5d134e3350a59139d67d6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_1b56eac4fe5d134e3350a59139d67d6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_1b56eac4fe5d134e3350a59139d67d6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_1b56eac4fe5d134e3350a59139d67d6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_1b56eac4fe5d134e3350a59139d67d6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_dba09ba03ffefda8f2726c71385338fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b
    def get_inputs(self):
        return [
            paddle.uniform([7581, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b9f8e1c85db4557fff30430af9b30c7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([7581, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_21d5152b9f03c359a771aa42b28c95e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce10a710f7565f96c00a1389a91699d4
    def get_inputs(self):
        return [
            paddle.uniform([3712, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3712, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5eb26b92970518f687a0bea510e387ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef457a8c2827717ad98c98ab5f8f38cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f3a508b496778ed0b643bc983d4abc57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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


class PrimitiveOp_654c5f6af86ec6b7bca9d326c0167e14(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 512, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_033ed509de49dffc0537d194bb79278b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_654c5f6af86ec6b7bca9d326c0167e14
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b03ab04a7c6a80550adca44ff1953292(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([20]),
            paddle.to_tensor([0.4588078260421753, 0.068813256919384, 0.27381643652915955, 0.0062551796436309814, 0.2485503852367401, 0.32595863938331604, 0.3125576674938202, 0.4546583294868469, 0.21912570297718048, 0.04495468735694885, 0.2998899519443512, 0.22375978529453278, 0.08261338621377945, 0.3015618324279785, 0.13809457421302795, 0.3192104697227478, 0.31224656105041504, 0.3313934803009033, 0.4789752960205078, 0.3796774744987488], dtype='float32').reshape([20]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_44c3ff04e72ab58ad3e9d4458ca255c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4588078260421753, 0.068813256919384, 0.27381643652915955, 0.0062551796436309814, 0.2485503852367401, 0.32595863938331604, 0.3125576674938202, 0.4546583294868469, 0.21912570297718048, 0.04495468735694885, 0.2998899519443512, 0.22375978529453278, 0.08261338621377945, 0.3015618324279785, 0.13809457421302795, 0.3192104697227478, 0.31224656105041504, 0.3313934803009033, 0.4789752960205078, 0.3796774744987488], dtype='float32').reshape([20]),
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([20]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ddf75c9f3c979bb9acc03625cbdf7fb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.03196848928928375], [0.24156440794467926], [0.0597226619720459], [0.32563695311546326]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.2075873762369156], [0.3073572814464569], [0.30402541160583496], [0.30721813440322876]], dtype='float32').reshape([4, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ef025180d9c70063a5865791baf4136f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06388451904058456], [0.14793279767036438], [0.12175875902175903], [0.26954853534698486]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.45928049087524414], [0.4224745035171509], [0.48921844363212585], [0.4449281692504883]], dtype='float32').reshape([4, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fdb1a6e2fd80014ccf6f505dd942966b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.11194626986980438], [0.3971475064754486], [0.26984596252441406], [0.32563695311546326]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.2075873762369156], [0.3073572814464569], [0.30402541160583496], [0.26391908526420593]], dtype='float32').reshape([4, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9cbed90ded4429f761c7fd1360f330ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06388451904058456], [0.2715681791305542], [0.12175875902175903], [0.30955228209495544]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.2821217179298401], [0.4224745035171509], [0.2759237587451935], [0.07581711560487747]], dtype='float32').reshape([4, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b09a7945f2d259abed9041b602fdaeb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.03196848928928375], [0.24156440794467926], [0.0597226619720459], [0.4864690601825714]], dtype='float32').reshape([4, 1]),
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
class TestPrimitiveOp_c310627930a62662838d5f5a20649c89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.17564170062541962], [0.14793279767036438], [0.27478042244911194], [0.26954853534698486]], dtype='float32').reshape([4, 1]),
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
class TestPrimitiveOp_2d9872f501403a66d0859da178ede8e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.034240152686834335], [-0.05106562748551369], [0.0040462748147547245], [-0.017011325806379318]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9bdff04d932a4faa51cdcafe984a839b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.11194626986980438], [0.3971475064754486], [0.26984596252441406], [0.4864690601825714]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.07909782230854034], [0.09549641609191895], [0.05401938408613205], [0.26391908526420593]], dtype='float32').reshape([4, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_54a438717cb3e1c8aad62a5de5b3bc7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.17564170062541962], [0.2715681791305542], [0.27478042244911194], [0.30955228209495544]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.2821217179298401], [0.4047701358795166], [0.2759237587451935], [0.07581711560487747]], dtype='float32').reshape([4, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d947b5bfd6646987f0ed9ea799d633d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0034977032337337732], [-0.0401805154979229], [-0.0002467622689437121], [0.052017755806446075]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.034240152686834335], [-0.05106562748551369], [0.0040462748147547245], [-0.017011325806379318]], dtype='float32').reshape([4, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_60dd8d8dd0a6c736859961e2164b950f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [-0.0], [0.0], [-0.0]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[10.789324760437012], [-0.2709052264690399], [17.3974609375], [1.3270291090011597]], dtype='float32').reshape([4, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2bdf48ae6973d8a4487566beb1fae511(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3c4fcffe104f1271b280c356c8da11fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce10a710f7565f96c00a1389a91699d4
    def get_inputs(self):
        return [
            paddle.uniform([2102, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2102, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ce661dac2ba540854aa6902991363736(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_ce661dac2ba540854aa6902991363736(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_ce661dac2ba540854aa6902991363736(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_ce661dac2ba540854aa6902991363736(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_ce661dac2ba540854aa6902991363736(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_ce661dac2ba540854aa6902991363736(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_ce661dac2ba540854aa6902991363736(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_ce661dac2ba540854aa6902991363736(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_ce661dac2ba540854aa6902991363736(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_ce661dac2ba540854aa6902991363736(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_ce661dac2ba540854aa6902991363736(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_15aa17eea687877cf988c12a2c059c6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b
    def get_inputs(self):
        return [
            paddle.uniform([4116, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_358bc8c1eb98b6e964028dcc0f56dec2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([4116, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3c4fcffe104f1271b280c356c8da11fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce10a710f7565f96c00a1389a91699d4
    def get_inputs(self):
        return [
            paddle.uniform([2102, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2102, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_033ed509de49dffc0537d194bb79278b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_654c5f6af86ec6b7bca9d326c0167e14
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_aae057c2bbb4508044ffb10c39a7e238(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef457a8c2827717ad98c98ab5f8f38cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_7182d24cb9ed83c3e793bde3926531e9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 6804, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_68b802329411cbd7dc7461a50172405a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7182d24cb9ed83c3e793bde3926531e9
    def get_inputs(self):
        return [
            paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6a08ef289f3e60926465a198a8ca4acc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.09660603851079941, 0.3473486304283142, 0.1792222410440445, 0.18537181615829468], [0.041187599301338196, 0.4844166040420532, 0.3308883607387543, 0.04537099227309227], [0.3677883446216583, 0.28455665707588196, 0.04476252570748329, 0.1025569960474968], [0.3677883446216583, 0.28455665707588196, 0.04476252570748329, 0.1025569960474968], [0.173814058303833, 0.032673005014657974, 0.2887442708015442, 0.17421869933605194]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([[0.3407200574874878, 0.21765947341918945, 0.25243550539016724, 0.42701151967048645], [0.23007768392562866, 0.04049093276262283, 0.40273621678352356, 0.1927415430545807], [0.10648475587368011, 0.16337205469608307, 0.42770418524742126, 0.41350576281547546], [0.10648475587368011, 0.16337205469608307, 0.42770418524742126, 0.41350576281547546], [0.4299686849117279, 0.23296520113945007, 0.4033758342266083, 0.018267221748828888]], dtype='float32').reshape([5, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_5eedd941575a8025b1f6142513fa4c9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_e1035ad1204a21003d0997b9a86a02a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce10a710f7565f96c00a1389a91699d4
    def get_inputs(self):
        return [
            paddle.uniform([4228, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4228, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4d474a5f3ac0252da65dc48c59432a88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_4d474a5f3ac0252da65dc48c59432a88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_4d474a5f3ac0252da65dc48c59432a88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_4d474a5f3ac0252da65dc48c59432a88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_4d474a5f3ac0252da65dc48c59432a88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_4d474a5f3ac0252da65dc48c59432a88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_4d474a5f3ac0252da65dc48c59432a88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_4d474a5f3ac0252da65dc48c59432a88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_4d474a5f3ac0252da65dc48c59432a88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_4d474a5f3ac0252da65dc48c59432a88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_4d474a5f3ac0252da65dc48c59432a88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20c42f8fe5fe3cbd282332085a2c3eae
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
class TestPrimitiveOp_fadf83c28010ecd5bd8925902fc6620e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b
    def get_inputs(self):
        return [
            paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0906bc96c54199c50c23766f6e035159(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e1035ad1204a21003d0997b9a86a02a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce10a710f7565f96c00a1389a91699d4
    def get_inputs(self):
        return [
            paddle.uniform([4228, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4228, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_46a383a87e08c137951d80a8dcedc762(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3381723165512085, 0.17756247520446777, 0.06380031257867813, 0.38561683893203735], [0.10080365091562271, 0.3514818549156189, 0.2887912094593048, 0.47205549478530884], [0.4350821375846863, 0.026866599917411804, 0.22386538982391357, 0.34845173358917236], [0.3381723165512085, 0.17756247520446777, 0.06380031257867813, 0.38561683893203735], [0.15402036905288696, 0.22495560348033905, 0.02383223921060562, 0.4998984634876251], [0.14966902136802673, 0.3276309669017792, 0.4284338355064392, 0.1937379390001297], [0.15402036905288696, 0.22495560348033905, 0.02383223921060562, 0.4998984634876251]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([[0.26299649477005005, 0.41595983505249023, 0.1868925839662552, 0.08783718943595886], [0.33425372838974, 0.35268688201904297, 0.04515431076288223, 0.4619998037815094], [0.30313247442245483, 0.06827733665704727, 0.1848611682653427, 0.08456997573375702], [0.26299649477005005, 0.41595983505249023, 0.1868925839662552, 0.08783718943595886], [0.3514348566532135, 0.43885645270347595, 0.2689119279384613, 0.44598138332366943], [0.4748520851135254, 0.467897891998291, 0.2615504264831543, 0.3895653486251831], [0.3514348566532135, 0.43885645270347595, 0.2689119279384613, 0.44598138332366943]], dtype='float32').reshape([7, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_5c1607c584dc7e104775c4f773344f40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([52, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([52, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0b42cee1f3c6b7f1a3c289d5c586e1b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19d0a8a77ea8ac0dd0fb5dcf3895371d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.3018275201320648]], [[0.33168932795524597]], [[0.46974313259124756]], [[0.010415264405310154]], [[0.16770502924919128]], [[0.08946999907493591]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.5554413795471191]], [[0.7320908904075623]], [[0.6382142901420593]], [[0.7114271521568298]], [[0.7790687680244446]], [[0.804796040058136]]], dtype='float32').reshape([6, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_84dbe9af23782d6554cfd945c82d3b06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19d0a8a77ea8ac0dd0fb5dcf3895371d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.3234318494796753]], [[0.10518792271614075]], [[0.29617762565612793]], [[0.2048494517803192]], [[0.140651136636734]], [[0.2106209695339203]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.7354300022125244]], [[0.5847243070602417]], [[0.7040197849273682]], [[0.5520831942558289]], [[0.7446761727333069]], [[0.70814448595047]]], dtype='float32').reshape([6, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_a929203e3767fdbf9f28631d4035808b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_7837eb9c20d5f70502fabaaf04eced81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_75aeed24fc04a1265ee35faaeb56814b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19d0a8a77ea8ac0dd0fb5dcf3895371d
    def get_inputs(self):
        return [
            paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_af7abc8f3c07a92290a9b6159b7d361b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8529a1be70d7f4c80d35871aebc24121(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_291a3baba83aebcfbb0959b81157d6f7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.4200737476348877, 0.2946609854698181]], [[0.47897306084632874, 0.29871895909309387]], [[0.34867048263549805, 0.2746659815311432]], [[0.022391103208065033, 0.1263965517282486]], [[0.00977354682981968, 0.026455778628587723]], [[0.39047884941101074, 0.10009819269180298]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.48737311363220215, 0.21872499585151672]], [[0.3569588363170624, 0.2996145188808441]], [[0.18173567950725555, 0.3620101511478424]], [[0.18104912340641022, 0.16540417075157166]], [[0.33887559175491333, 0.28572165966033936]], [[0.4271915853023529, 0.10836858302354813]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f7ca0670ac8140f9abb4a9bbf7b9ada2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_291a3baba83aebcfbb0959b81157d6f7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.08230464160442352, 0.17765505611896515]], [[0.41948506236076355, 0.2993577718734741]], [[0.26732832193374634, 0.14748147130012512]], [[0.4790646433830261, 0.15387652814388275]], [[0.47866836190223694, 0.19755837321281433]], [[0.2591554522514343, 0.37071239948272705]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.48737311363220215, 0.21872499585151672]], [[0.3569588363170624, 0.2996145188808441]], [[0.18173567950725555, 0.3620101511478424]], [[0.18104912340641022, 0.16540417075157166]], [[0.33887559175491333, 0.28572165966033936]], [[0.4271915853023529, 0.10836858302354813]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_73f756389c5d36834576d0da44355b96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_291a3baba83aebcfbb0959b81157d6f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 21824, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.22964154183864594, 0.24316290020942688]], [[0.3824300467967987, 0.4855898916721344]], [[0.09435852617025375, 0.4004281461238861]], [[0.01536477915942669, 0.1013849675655365]], [[0.02149014361202717, 0.3536393940448761]], [[0.28038325905799866, 0.15163615345954895]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0b543be2087ce063597a56b11ea6ac5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([16]),
            paddle.to_tensor([0.021799949929118156, 0.0553545206785202, 0.1331569254398346, 0.12268710136413574, 0.39009812474250793, 0.20741882920265198, 0.16273048520088196, 0.31630992889404297, 0.1374664157629013, 0.005575885064899921, 0.18881046772003174, 0.18218941986560822, 0.18066896498203278, 0.39684590697288513, 0.053641725331544876, 0.24692824482917786], dtype='float32').reshape([16]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e38e3fde4ad5ba06b82555dd295fc739(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.021799949929118156, 0.0553545206785202, 0.1331569254398346, 0.12268710136413574, 0.39009812474250793, 0.20741882920265198, 0.16273048520088196, 0.31630992889404297, 0.1374664157629013, 0.005575885064899921, 0.18881046772003174, 0.18218941986560822, 0.18066896498203278, 0.39684590697288513, 0.053641725331544876, 0.24692824482917786], dtype='float32').reshape([16]),
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([16]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_372aad51121867288b938e82c520cce6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_372aad51121867288b938e82c520cce6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0388175443d5f0af0087911a194535d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_6e02c6ca3972717e5b5b70e1ebc41e36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_aedf9ebd4443f497cef15d11686f58c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1778, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1778, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_581fbc8b4af7af90eb98b9eecfafda24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_581fbc8b4af7af90eb98b9eecfafda24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_581fbc8b4af7af90eb98b9eecfafda24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_581fbc8b4af7af90eb98b9eecfafda24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_581fbc8b4af7af90eb98b9eecfafda24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_581fbc8b4af7af90eb98b9eecfafda24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_581fbc8b4af7af90eb98b9eecfafda24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_581fbc8b4af7af90eb98b9eecfafda24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_581fbc8b4af7af90eb98b9eecfafda24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_581fbc8b4af7af90eb98b9eecfafda24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_581fbc8b4af7af90eb98b9eecfafda24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_52d6c0904952e02947e994d334c88237(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b
    def get_inputs(self):
        return [
            paddle.uniform([3549, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_38dc692e168c69c6d9272c552d7727e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([3549, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_aedf9ebd4443f497cef15d11686f58c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1778, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1778, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eecae8933709bf2506de121193ed0b13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4083593487739563, 0.0464426651597023, 0.19350320100784302, 0.0735485628247261], [0.04275652766227722, 0.42302489280700684, 0.1234317496418953, 0.4487559199333191], [0.3989906907081604, 0.423295795917511, 0.035104915499687195, 0.2857874929904938], [0.44080936908721924, 0.3033484220504761, 0.12672246992588043, 0.06597451865673065], [0.2818484306335449, 0.34505268931388855, 0.3520521819591522, 0.14433987438678741]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([[0.2586134672164917, 0.15547722578048706, 0.1751316338777542, 0.03089195117354393], [0.07368368655443192, 0.2442678064107895, 0.004473511129617691, 0.24331246316432953], [0.23028400540351868, 0.4056583642959595, 0.3727683126926422, 0.28531497716903687], [0.2434556633234024, 0.2522284686565399, 0.04619003087282181, 0.0900825560092926], [0.21415281295776367, 0.2867811918258667, 0.02609613910317421, 0.35222798585891724]], dtype='float32').reshape([5, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_753e4f9951039e8fb6243310705899f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19d0a8a77ea8ac0dd0fb5dcf3895371d
    def get_inputs(self):
        return [
            paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_536ed0de413ec3afb4ee079acceb8755(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.11813344806432724, 0.3100629150867462, 0.1310870349407196, 0.048055391758680344], [0.26968440413475037, 0.038270726799964905, 0.2786773145198822, 0.06464090198278427], [0.48029011487960815, 0.10376059263944626, 0.34819114208221436, 0.10710859298706055], [0.26968440413475037, 0.038270726799964905, 0.2786773145198822, 0.06464090198278427], [0.48029011487960815, 0.10376059263944626, 0.34819114208221436, 0.10710859298706055]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([[0.10199055820703506, 0.25440630316734314, 0.3121725618839264, 0.12423137575387955], [0.03548659756779671, 0.14141957461833954, 0.2776472568511963, 0.0903051495552063], [0.20749840140342712, 0.26365041732788086, 0.04141669720411301, 0.44134292006492615], [0.03548659756779671, 0.14141957461833954, 0.2776472568511963, 0.0903051495552063], [0.20749840140342712, 0.26365041732788086, 0.04141669720411301, 0.44134292006492615]], dtype='float32').reshape([5, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_b4e9b28607eff013e91bf04e8b23e889(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.10817418247461319], [0.21830850839614868], [0.1470475047826767], [0.2576325237751007], [0.06852712482213974], [0.39306512475013733], [0.006903389468789101], [0.004259239882230759], [0.36045563220977783]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.3870587646961212], [0.0602361299097538], [0.3039933145046234], [0.4867640435695648], [0.405436635017395], [0.3102510869503021], [0.4106326401233673], [0.37027329206466675], [0.4077361524105072]], dtype='float32').reshape([9, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e32c3fe8cf74a7896c71a84ee6048163(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.013448690064251423], [0.1428917944431305], [0.18482916057109833], [0.2410021275281906], [0.11816191673278809], [0.2892489433288574], [0.022538647055625916], [0.42789679765701294], [0.11527103185653687]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.3498869836330414], [0.26301130652427673], [0.3965403735637665], [0.37756359577178955], [0.17188876867294312], [0.06343194842338562], [0.4906500577926636], [0.49394193291664124], [0.43974825739860535]], dtype='float32').reshape([9, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c80fa5ee0ab2b73d17a44c009b818b00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.10817418247461319], [0.21830850839614868], [0.1470475047826767], [0.2576325237751007], [0.06852712482213974], [0.39306512475013733], [0.006903389468789101], [0.1592971235513687], [0.36045563220977783]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.16345298290252686], [0.03933706879615784], [0.26740995049476624], [0.15774424374103546], [0.22924834489822388], [0.3102510869503021], [0.4106326401233673], [0.31178075075149536], [0.4077361524105072]], dtype='float32').reshape([9, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6889ab982fee1064fb7a50e59c167b7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.013448690064251423], [0.39393821358680725], [0.41981005668640137], [0.2410021275281906], [0.28984448313713074], [0.2892489433288574], [0.022538647055625916], [0.48357874155044556], [0.11527103185653687]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.22238902747631073], [0.048635415732860565], [0.2709272503852844], [0.37756359577178955], [0.17188876867294312], [0.06343194842338562], [0.4906500577926636], [0.49394193291664124], [0.19016608595848083]], dtype='float32').reshape([9, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6dc561b3ed11d1f3c93fdc09d549e00a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3560943007469177], [0.2843972146511078], [0.19924697279930115], [0.39769965410232544], [0.1095288023352623], [0.4948180615901947], [0.43056219816207886], [0.004259239882230759], [0.4542475640773773]], dtype='float32').reshape([9, 1]),
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
class TestPrimitiveOp_32f379606fe67497f0bc134f5b82a199(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.09306798875331879], [0.1428917944431305], [0.18482916057109833], [0.4609587788581848], [0.11816191673278809], [0.3480241000652313], [0.3102295994758606], [0.42789679765701294], [0.15572763979434967]], dtype='float32').reshape([9, 1]),
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
class TestPrimitiveOp_dc4b3ba0bb9b026c01f18d05a51dfaf6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.01950223371386528], [0.03487321734428406], [0.0042560771107673645], [-0.029554974287748337], [-0.04360903054475784], [0.10207147896289825], [0.17654693126678467], [0.012095225043594837], [-0.09465660154819489]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.018700817599892616], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4675f232b75aa7c42c135c5f5ad91506(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3560943007469177], [0.2843972146511078], [0.19924697279930115], [0.39769965410232544], [0.1095288023352623], [0.4948180615901947], [0.43056219816207886], [0.1592971235513687], [0.4542475640773773]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.16345298290252686], [0.03933706879615784], [0.26740995049476624], [0.15774424374103546], [0.22924834489822388], [0.246979221701622], [0.08601068705320358], [0.31178075075149536], [0.10850615799427032]], dtype='float32').reshape([9, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d9cc7f9c0e5e0d12a06b2dac07ee6408(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.09306798875331879], [0.39393821358680725], [0.41981005668640137], [0.4609587788581848], [0.28984448313713074], [0.3480241000652313], [0.3102295994758606], [0.48357874155044556], [0.15572763979434967]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.22238902747631073], [0.048635415732860565], [0.2709272503852844], [0.2822781205177307], [0.03485541045665741], [0.011633493937551975], [0.3463441729545593], [0.45662522315979004], [0.19016608595848083]], dtype='float32').reshape([9, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_585781309cf679830c4742d8a8168ae6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.024912575259804726], [0.08461995422840118], [-0.01014829520136118], [0.042875390499830246], [-0.030527176335453987], [0.08337066322565079], [-0.012443330138921738], [-0.004109970293939114], [-0.011906796135008335]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.01950223371386528], [0.03487321734428406], [0.0042560771107673645], [-0.029554974287748337], [-0.04360903054475784], [0.08337066322565079], [0.17654693126678467], [0.012095225043594837], [-0.09465660154819489]], dtype='float32').reshape([9, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3ab043f9db6ded50b730a864a28ff48f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [-0.0], [-0.0], [0.22430932521820068], [0.0], [0.0], [-0.0]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[1.7828270196914673], [0.587884247303009], [1.4193884134292603], [1.689322590827942], [-0.4285314083099365], [0.0], [15.188077926635742], [3.9428985118865967], [-6.949796199798584]], dtype='float32').reshape([9, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1b2649627c56cc22f68d3cbea932371b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19d0a8a77ea8ac0dd0fb5dcf3895371d
    def get_inputs(self):
        return [
            paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_240608e199211c86bcbc0afed5a46001(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19d0a8a77ea8ac0dd0fb5dcf3895371d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.493002325296402]], [[0.021986098960042]], [[0.009986251592636108]], [[0.47580042481422424]], [[0.050395287573337555]], [[0.026547249406576157]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.5732123851776123]], [[0.7980857491493225]], [[0.6817842125892639]], [[0.7067649364471436]], [[0.592467725276947]], [[0.786468505859375]]], dtype='float32').reshape([6, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2bc7df90204aca0f4e5649e822f2f64f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19d0a8a77ea8ac0dd0fb5dcf3895371d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.38881805539131165]], [[0.16338205337524414]], [[0.16764523088932037]], [[0.06548044085502625]], [[0.20316286385059357]], [[0.42358702421188354]]], dtype='float32').reshape([6, 1, 1]),
            paddle.to_tensor([[[0.6939191818237305]], [[0.6774009466171265]], [[0.5296899080276489]], [[0.6363528966903687]], [[0.6614072918891907]], [[0.5798671245574951]]], dtype='float32').reshape([6, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f4efdd4f4553731eda82c2570b8cda74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_291a3baba83aebcfbb0959b81157d6f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_07c09ca13e6b46d6220124c31783cce1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([5640, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([5640, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_711e7314d14e43609014bfb8b7d566b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_711e7314d14e43609014bfb8b7d566b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_711e7314d14e43609014bfb8b7d566b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_711e7314d14e43609014bfb8b7d566b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_711e7314d14e43609014bfb8b7d566b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_711e7314d14e43609014bfb8b7d566b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_711e7314d14e43609014bfb8b7d566b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_711e7314d14e43609014bfb8b7d566b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_711e7314d14e43609014bfb8b7d566b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_711e7314d14e43609014bfb8b7d566b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_711e7314d14e43609014bfb8b7d566b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_03b1906bab8cf1f48b4606ade5191acd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b
    def get_inputs(self):
        return [
            paddle.uniform([11109, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4f0222eca32e6403f4049a2fb3b688ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([11109, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_07c09ca13e6b46d6220124c31783cce1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([5640, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([5640, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b8813d1ff55826a67087c1015e0825b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.40535181760787964, 0.18487544357776642, 0.12082479894161224, 0.15880312025547028], [0.20091252028942108, 0.08696173131465912, 0.253089040517807, 0.31328579783439636], [0.25166910886764526, 0.34895312786102295, 0.10745351016521454, 0.08950001746416092], [0.20091252028942108, 0.08696173131465912, 0.253089040517807, 0.31328579783439636], [0.25166910886764526, 0.34895312786102295, 0.10745351016521454, 0.08950001746416092], [0.11380904167890549, 0.49813464283943176, 0.28097593784332275, 0.20141464471817017], [0.11380904167890549, 0.49813464283943176, 0.28097593784332275, 0.20141464471817017]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([[0.06191736087203026, 0.3354967534542084, 0.11539153009653091, 0.05804868042469025], [0.3243649899959564, 0.2906089127063751, 0.09762857109308243, 0.3227904438972473], [0.48022186756134033, 0.03066052496433258, 0.0535271018743515, 0.20987284183502197], [0.3243649899959564, 0.2906089127063751, 0.09762857109308243, 0.3227904438972473], [0.48022186756134033, 0.03066052496433258, 0.0535271018743515, 0.20987284183502197], [0.09828855842351913, 0.20395898818969727, 0.08289018273353577, 0.36240696907043457], [0.09828855842351913, 0.20395898818969727, 0.08289018273353577, 0.36240696907043457]], dtype='float32').reshape([7, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0e93be1b02e7657a17ee401eb2ea5cf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0e93be1b02e7657a17ee401eb2ea5cf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6a0d88c3b31ac0ec78be6d8038ebdb82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_18b385f2773e20549fa68a071089e091(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a61d16187b170698f9108434bcba5729(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.21110419929027557, 0.35089540481567383, 0.2971991300582886, 0.16651640832424164, 0.40372434258461, 0.39073458313941956], dtype='float32').reshape([6]),
            paddle.to_tensor([0.4102131128311157, 0.047395769506692886, 0.13399538397789001, 0.29094526171684265, 0.009467806667089462, 0.18842779099941254], dtype='float32').reshape([6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_edbc75489aa0d60ff9ffddf81b5d417f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.038237329572439194, 0.16254805028438568, 0.004502696916460991, 0.1902225911617279, 0.19850163161754608, 0.19214455783367157], dtype='float32').reshape([6]),
            paddle.to_tensor([0.1450636386871338, 0.25107648968696594, 0.23796823620796204, 0.386831134557724, 0.3538130819797516, 0.40666136145591736], dtype='float32').reshape([6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7b795e41a4c1fbfd6bdca0dabd29e280(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3418285846710205, 0.42906653881073, 0.04830992594361305, 0.33282768726348877, 0.28770944476127625, 0.12256492674350739], dtype='float32').reshape([6]),
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
class TestPrimitiveOp_29b20e447022bdff8e44bd46f99956f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4553065598011017, 0.16179899871349335, 0.38409265875816345, 0.37823209166526794, 0.21884433925151825, 0.2555798888206482], dtype='float32').reshape([6]),
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
class TestPrimitiveOp_924d332831e3e6d2743abd4f612bfae5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3418285846710205, 0.35089540481567383, 0.04830992594361305, 0.29094526171684265, 0.28770944476127625, 0.12256492674350739], dtype='float32').reshape([6]),
            paddle.to_tensor([0.4102131128311157, 0.20509734749794006, 0.16082343459129333, 0.4748803675174713, 0.07652277499437332, 0.20391766726970673], dtype='float32').reshape([6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f65bc9f9e3b8dce8341ff33f559d2788(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1450636386871338, 0.16179899871349335, 0.23796823620796204, 0.37823209166526794, 0.21884433925151825, 0.2555798888206482], dtype='float32').reshape([6]),
            paddle.to_tensor([0.4793972969055176, 0.25107648968696594, 0.23796823620796204, 0.386831134557724, 0.3538130819797516, 0.48301398754119873], dtype='float32').reshape([6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_429ff7dce7d8b733136f0037a2e1b008(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4102131128311157, 0.35089540481567383, 0.2971991300582886, 0.29094526171684265, 0.40372434258461, 0.39073458313941956], dtype='float32').reshape([6]),
            paddle.to_tensor([0.4102131128311157, 0.047395769506692886, 0.13399538397789001, 0.29094526171684265, 0.009467806667089462, 0.18842779099941254], dtype='float32').reshape([6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7b9fa4029744a810baeb3f5f0ca16f31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1450636386871338, 0.25107648968696594, 0.23796823620796204, 0.386831134557724, 0.3538130819797516, 0.40666136145591736], dtype='float32').reshape([6]),
            paddle.to_tensor([0.1450636386871338, 0.25107648968696594, 0.23796823620796204, 0.386831134557724, 0.3538130819797516, 0.40666136145591736], dtype='float32').reshape([6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_570b6914e2aba624653da6012b0796ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.0001800392201403156, 0.0017102111596614122, -0.033292002975940704, -0.01555620227009058, 0.04252384603023529, 0.018502388149499893], dtype='float32').reshape([6]),
            paddle.to_tensor([0.0, -0.0, -0.0, 0.0, -0.0, 0.0], dtype='float32').reshape([6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_27aab46b52317ccf07492f1442f7946f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.31065866351127625, 0.1991455852985382, 0.2155972570180893, 0.22873082756996155, 0.20659607648849487, 0.28958117961883545], dtype='float32').reshape([6]),
            paddle.to_tensor([0.3380919098854065, 0.31708192825317383, 0.10456667840480804, 0.40385401248931885, 0.18211610615253448, 0.16324129700660706], dtype='float32').reshape([6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a090f43ee249e6724a0d17519a81fe5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.09165048599243164, 0.20681226253509521, 0.12123546749353409, 0.28852686285972595, 0.27615734934806824, 0.29940295219421387], dtype='float32').reshape([6]),
            paddle.to_tensor([0.46735191345214844, 0.1579810380935669, 0.2361459732055664, 0.32347702980041504, 0.1181660071015358, 0.36929693818092346], dtype='float32').reshape([6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ed0a81c75952d9adc75ed473b849e8fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4102131128311157, 0.42906653881073, 0.2971991300582886, 0.33282768726348877, 0.40372434258461, 0.39073458313941956], dtype='float32').reshape([6]),
            paddle.to_tensor([0.3343552052974701, 0.047395769506692886, 0.13399538397789001, 0.29094526171684265, 0.009467806667089462, 0.18842779099941254], dtype='float32').reshape([6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_94a4953afe90942d79d977cf7910b97b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4553065598011017, 0.25107648968696594, 0.38409265875816345, 0.386831134557724, 0.3538130819797516, 0.40666136145591736], dtype='float32').reshape([6]),
            paddle.to_tensor([0.1450636386871338, 0.15416307747364044, 0.08819928020238876, 0.2687219977378845, 0.017487674951553345, 0.40666136145591736], dtype='float32').reshape([6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6ae434afba7a1bab2098e08b8ab64305(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.3008045256137848, 1.5367159843444824, -0.36336562037467957, -0.9140454530715942, 0.8092215657234192, 0.3435162603855133], dtype='float32').reshape([6]),
            paddle.to_tensor([1.078359842300415, -1.286978840827942, -0.6100871562957764, 0.5642430186271667, -1.1955293416976929, -0.7561135292053223], dtype='float32').reshape([6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_137aea93ef6c6d10ce5501ce304140e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1797, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1797, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8fb64ebd54e6ea2d6ee411e13dec94a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_8fb64ebd54e6ea2d6ee411e13dec94a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_8fb64ebd54e6ea2d6ee411e13dec94a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_8fb64ebd54e6ea2d6ee411e13dec94a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_8fb64ebd54e6ea2d6ee411e13dec94a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_8fb64ebd54e6ea2d6ee411e13dec94a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_8fb64ebd54e6ea2d6ee411e13dec94a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_8fb64ebd54e6ea2d6ee411e13dec94a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_8fb64ebd54e6ea2d6ee411e13dec94a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_8fb64ebd54e6ea2d6ee411e13dec94a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_8fb64ebd54e6ea2d6ee411e13dec94a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_52d6c0904952e02947e994d334c88237(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b
    def get_inputs(self):
        return [
            paddle.uniform([3549, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_38dc692e168c69c6d9272c552d7727e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([3549, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_137aea93ef6c6d10ce5501ce304140e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1797, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1797, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e426acc2a8a3c866ade65ea71932622b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19d0a8a77ea8ac0dd0fb5dcf3895371d
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_958bd85215449bcd203b84066797db5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([24]),
            paddle.to_tensor([0.14985322952270508, 0.3614925444126129, 0.39877936244010925, 0.053330495953559875, 0.13969476521015167, 0.0027347251307219267, 0.28718096017837524, 0.06502088159322739, 0.490936279296875, 0.3843059539794922, 0.1196126788854599, 0.15985898673534393, 0.024629564955830574, 0.3084601163864136, 0.15704958140850067, 0.3779698312282562, 0.4783381223678589, 0.23180651664733887, 0.2301483005285263, 0.21951468288898468, 0.4981519877910614, 0.11176123470067978, 0.3388000726699829, 0.1751078963279724], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_359d7a62bab0a45964a45ed24d1f585c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.14985322952270508, 0.3614925444126129, 0.39877936244010925, 0.053330495953559875, 0.13969476521015167, 0.0027347251307219267, 0.28718096017837524, 0.06502088159322739, 0.490936279296875, 0.3843059539794922, 0.1196126788854599, 0.15985898673534393, 0.024629564955830574, 0.3084601163864136, 0.15704958140850067, 0.3779698312282562, 0.4783381223678589, 0.23180651664733887, 0.2301483005285263, 0.21951468288898468, 0.4981519877910614, 0.11176123470067978, 0.3388000726699829, 0.1751078963279724], dtype='float32').reshape([24]),
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_367efeaffffb39edf014e07940175ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_c19782b9171af293d660dd3abb7b129a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_19a75ac0570763cc962d8cddfc43662c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1501, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1501, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ffc80463e0d01b837ca44ba1735a5eda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_ffc80463e0d01b837ca44ba1735a5eda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_ffc80463e0d01b837ca44ba1735a5eda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_ffc80463e0d01b837ca44ba1735a5eda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_ffc80463e0d01b837ca44ba1735a5eda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_ffc80463e0d01b837ca44ba1735a5eda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_ffc80463e0d01b837ca44ba1735a5eda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_ffc80463e0d01b837ca44ba1735a5eda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_ffc80463e0d01b837ca44ba1735a5eda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_ffc80463e0d01b837ca44ba1735a5eda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_ffc80463e0d01b837ca44ba1735a5eda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_3ed75efcd84ec090d0b5f8a4bdcffa95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b
    def get_inputs(self):
        return [
            paddle.uniform([3024, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a1d0e3e7e00154ab42eff6103c79fdb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([3024, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_19a75ac0570763cc962d8cddfc43662c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1501, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1501, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_82617b9af541ee665e20f7afbba3a185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_95329544739f16b1ba6257c58390f2ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([4]),
            paddle.to_tensor([0.31062570214271545, 0.34576329588890076, 0.1544123739004135, 0.4026666283607483], dtype='float32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0344c476ae89de1a5be9f95cb2f83303(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.31062570214271545, 0.34576329588890076, 0.1544123739004135, 0.4026666283607483], dtype='float32').reshape([4]),
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_4d28e76b50ed241aed2c5946f80248f2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 - input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_88abd78c6b8823b63b747f20ccf526af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d28e76b50ed241aed2c5946f80248f2
    def get_inputs(self):
        return [
            paddle.to_tensor([4], dtype='int32').reshape([1]),
            paddle.to_tensor([2], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6de5048b4b9678ffe6cdcce96f01d92f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d28e76b50ed241aed2c5946f80248f2
    def get_inputs(self):
        return [
            paddle.to_tensor([7], dtype='int32').reshape([1]),
            paddle.to_tensor([3], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_180f6ba1845e2f17d908d9b250386175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_f38049eed6836134ebfa696e207c5ec1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.10230942815542221, 0.4656737744808197, 0.23023895919322968, 0.3134593963623047], [0.037858448922634125, 0.4459499418735504, 0.04106490686535835, 0.34768176078796387], [0.3960135579109192, 0.30466198921203613, 0.1388903111219406, 0.2818836569786072], [0.43273648619651794, 0.2460307776927948, 0.4382016062736511, 0.267194002866745], [0.43273648619651794, 0.2460307776927948, 0.4382016062736511, 0.267194002866745], [0.3960135579109192, 0.30466198921203613, 0.1388903111219406, 0.2818836569786072]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([[0.036272622644901276, 0.019473768770694733, 0.2123793065547943, 0.2291458547115326], [0.31556394696235657, 0.4416036903858185, 0.21069373190402985, 0.10256848484277725], [0.19106312096118927, 0.24031859636306763, 0.019208991900086403, 0.38589227199554443], [0.49473837018013, 0.3024352490901947, 0.3373301327228546, 0.3158513307571411], [0.49473837018013, 0.3024352490901947, 0.3373301327228546, 0.3158513307571411], [0.19106312096118927, 0.24031859636306763, 0.019208991900086403, 0.38589227199554443]], dtype='float32').reshape([6, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0226ae1e09d1d8c97bd533dc8f9c09a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2260180413722992, 0.19949793815612793, 0.21764975786209106, 0.3027788996696472], [0.14532725512981415, 0.257521390914917, 0.46131858229637146, 0.07517838478088379], [0.1859636753797531, 0.011614393442869186, 0.09635768085718155, 0.33755823969841003], [0.017686065286397934, 0.058734674006700516, 0.36337247490882874, 0.492013156414032], [0.2260180413722992, 0.19949793815612793, 0.21764975786209106, 0.3027788996696472]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([[0.3445029556751251, 0.45443597435951233, 0.2382742315530777, 0.03435212001204491], [0.10079831629991531, 0.2235482782125473, 0.2061411738395691, 0.343816339969635], [0.35300323367118835, 0.1610371619462967, 0.1562039703130722, 0.1116558089852333], [0.19510143995285034, 0.2744866907596588, 0.15754464268684387, 0.30516183376312256], [0.3445029556751251, 0.45443597435951233, 0.2382742315530777, 0.03435212001204491]], dtype='float32').reshape([5, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_56bd3d7f1023b32564a39a9793b4d622(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_55830e6682332d190b9d1c2ce9af6c22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.16136038303375244]], dtype='float32').reshape([1, 1]),
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
class TestPrimitiveOp_2f6dd69d4ca2c39b86352cb3f1e583f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.43908387422561646]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.235838383436203]], dtype='float32').reshape([1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7610e93fa1ad6e4bfd84bcd7f5be0c05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.26837071776390076]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.08844736218452454]], dtype='float32').reshape([1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2f6dd69d4ca2c39b86352cb3f1e583f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.43908387422561646]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.235838383436203]], dtype='float32').reshape([1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_55830e6682332d190b9d1c2ce9af6c22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.16136038303375244]], dtype='float32').reshape([1, 1]),
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
class TestPrimitiveOp_e9881d22393f4a735ae67492585cfb50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4695817530155182]], dtype='float32').reshape([1, 1]),
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
class TestPrimitiveOp_3e48a0c956cc032672e5bd9727a2e8b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.04066997021436691]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7610e93fa1ad6e4bfd84bcd7f5be0c05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.26837071776390076]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.08844736218452454]], dtype='float32').reshape([1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e9881d22393f4a735ae67492585cfb50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4695817530155182]], dtype='float32').reshape([1, 1]),
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
class TestPrimitiveOp_67fc21e1adfd0ed06b07f914805dafd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.04341842234134674]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.04066997021436691]], dtype='float32').reshape([1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1a9003ea910d2a73a12256a9f2d8bb55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[1.9366984367370605]], dtype='float32').reshape([1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_db2a0e654bd120967d8c03b1c7def925(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.35055011510849], [0.09527037292718887], [0.32676854729652405], [0.12946005165576935], [0.37342819571495056], [0.03219883143901825]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.3338686227798462], [0.07760456204414368], [0.3587840497493744], [0.40559735894203186], [0.3328937888145447], [0.46134132146835327]], dtype='float32').reshape([6, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2a2645d16c91edb48b527d9c93281b3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.405924528837204], [0.21171081066131592], [0.1269102543592453], [0.3708653151988983], [0.14785419404506683], [0.3854176700115204]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.39591267704963684], [0.25403302907943726], [0.3512172996997833], [0.486686110496521], [0.46994784474372864], [0.44575998187065125]], dtype='float32').reshape([6, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_915e5387cd386f549cba4fec6cc8be82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.35055011510849], [0.09527037292718887], [0.40782997012138367], [0.12946005165576935], [0.37342819571495056], [0.3238355815410614]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.3338686227798462], [0.0581473633646965], [0.27728891372680664], [0.40559735894203186], [0.2994241714477539], [0.46134132146835327]], dtype='float32').reshape([6, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ef4aad64752b01997bb2010e321de425(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.405924528837204], [0.31001538038253784], [0.2668556869029999], [0.47325971722602844], [0.4950177073478699], [0.3854176700115204]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.39591267704963684], [0.11558108776807785], [0.3512172996997833], [0.486686110496521], [0.46994784474372864], [0.3137425482273102]], dtype='float32').reshape([6, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6ed50df2234a689874f558a0eeb37521(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.355660617351532], [0.35937029123306274], [0.32676854729652405], [0.3721979558467865], [0.4757102429866791], [0.03219883143901825]], dtype='float32').reshape([6, 1]),
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
class TestPrimitiveOp_b38b84a5259e83db24292e12958eaf6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.43543845415115356], [0.21171081066131592], [0.1269102543592453], [0.3708653151988983], [0.14785419404506683], [0.4037357568740845]], dtype='float32').reshape([6, 1]),
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
class TestPrimitiveOp_25d4f96fc7acde48b7c5d2971da27b74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.09359443187713623], [-0.004706964362412691], [-0.01218957919627428], [-0.004550507292151451], [-0.022184086963534355], [0.0035779764875769615]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.00016701262211427093], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_be31ea024f01e1a84533aad0db482915(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.355660617351532], [0.35937029123306274], [0.40782997012138367], [0.3721979558467865], [0.4757102429866791], [0.3238355815410614]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.06195531785488129], [0.0581473633646965], [0.27728891372680664], [0.39517515897750854], [0.2994241714477539], [0.35186487436294556]], dtype='float32').reshape([6, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c516f20c732ca49b97c315878be9b0ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.43543845415115356], [0.31001538038253784], [0.2668556869029999], [0.47325971722602844], [0.4950177073478699], [0.4037357568740845]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.11733926832675934], [0.11558108776807785], [0.09014913439750671], [0.011464092880487442], [0.316177636384964], [0.3137425482273102]], dtype='float32').reshape([6, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f2fe03fd2101fb0041be836e9ec1cf2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.09342741966247559], [0.058568064123392105], [0.02306745946407318], [-0.010610772296786308], [0.03152701258659363], [-0.002522445982322097]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.09342741966247559], [-0.004706964362412691], [-0.01218957919627428], [-0.004550507292151451], [-0.022184086963534355], [0.0035779764875769615]], dtype='float32').reshape([6, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_771057f4b138ec75b770be8019c2538a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0017876189667731524], [-0.0], [-0.0], [-0.0], [-0.0], [0.0]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.0], [1.0803674459457397], [1.52843177318573], [0.5711426734924316], [1.703653335571289], [2.418455123901367]], dtype='float32').reshape([6, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c8cd9d9c29e747e89cc21bc77f32ee53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4666564464569092, 0.3648544251918793, 0.18811792135238647, 0.14878439903259277], [0.4952784776687622, 0.14019908010959625, 0.14755922555923462, 0.26648133993148804], [0.45726096630096436, 0.4442957639694214, 0.05174024775624275, 0.17932143807411194], [0.355541855096817, 0.4766886830329895, 0.16455477476119995, 0.24976997077465057]], dtype='float32').reshape([4, 4]),
            paddle.to_tensor([[0.2993108332157135, 0.2873929440975189, 0.3280841112136841, 0.0652041807770729], [0.004542017355561256, 0.09385919570922852, 0.0289857666939497, 0.4194986820220947], [0.3477599620819092, 0.3219774067401886, 0.10543779283761978, 0.32350409030914307], [0.026139285415410995, 0.1471622884273529, 0.12444959580898285, 0.19544094800949097]], dtype='float32').reshape([4, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_34d594e2342292a8df4dd93baf321763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_062112f4306a0eb96a52aef08f1165b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_ad01ba209aa563c77d8a0f5ecf6c7ca2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1a949e17a9dfaa427065d95f88b44a55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([2078, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2078, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a28df47a9be182978c8674f4981a68b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_a28df47a9be182978c8674f4981a68b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_a28df47a9be182978c8674f4981a68b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_a28df47a9be182978c8674f4981a68b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_a28df47a9be182978c8674f4981a68b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_a28df47a9be182978c8674f4981a68b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_a28df47a9be182978c8674f4981a68b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_a28df47a9be182978c8674f4981a68b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_a28df47a9be182978c8674f4981a68b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_a28df47a9be182978c8674f4981a68b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_a28df47a9be182978c8674f4981a68b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_15aa17eea687877cf988c12a2c059c6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b
    def get_inputs(self):
        return [
            paddle.uniform([4116, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_358bc8c1eb98b6e964028dcc0f56dec2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([4116, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1a949e17a9dfaa427065d95f88b44a55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([2078, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2078, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a39b70f293f0a1c4860e458c41b36f2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.05868971347808838, 0.0931183323264122, 0.4817720353603363, 0.2228946089744568], [0.05868971347808838, 0.0931183323264122, 0.4817720353603363, 0.2228946089744568], [0.1748245358467102, 0.4562341272830963, 0.3886791169643402, 0.12167710065841675], [0.04895717278122902, 0.004515435546636581, 0.1815401315689087, 0.42079052329063416], [0.4375079870223999, 0.014157120138406754, 0.24924929440021515, 0.234329953789711], [0.17357571423053741, 0.2881641089916229, 0.16192957758903503, 0.2890987992286682], [0.18074443936347961, 0.18766231834888458, 0.4655025601387024, 0.06498397141695023]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([[0.427163302898407, 0.26374343037605286, 0.46053290367126465, 0.00455693993717432], [0.427163302898407, 0.26374343037605286, 0.46053290367126465, 0.00455693993717432], [0.030603915452957153, 0.29073208570480347, 0.43393781781196594, 0.03389497473835945], [0.2495403289794922, 0.18423983454704285, 0.046639569103717804, 0.4862843453884125], [0.23021572828292847, 0.21562013030052185, 0.24692431092262268, 0.44247740507125854], [0.20590561628341675, 0.06558937579393387, 0.27467429637908936, 0.04392644390463829], [0.4704324007034302, 0.39693382382392883, 0.07851158082485199, 0.20433372259140015]], dtype='float32').reshape([7, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_40f637a1c203bb54d195c3251937c47a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_691fd97a65d1ec77739972ae555392e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_291a3baba83aebcfbb0959b81157d6f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_831765e2b98b9a37fe2d11e9ecd4a5b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([4653, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4653, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_472345c9d4b8b58fea1f6be3719e374d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_472345c9d4b8b58fea1f6be3719e374d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_472345c9d4b8b58fea1f6be3719e374d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_472345c9d4b8b58fea1f6be3719e374d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_472345c9d4b8b58fea1f6be3719e374d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_472345c9d4b8b58fea1f6be3719e374d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_472345c9d4b8b58fea1f6be3719e374d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_472345c9d4b8b58fea1f6be3719e374d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_472345c9d4b8b58fea1f6be3719e374d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_472345c9d4b8b58fea1f6be3719e374d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_472345c9d4b8b58fea1f6be3719e374d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_fcb8107f4be5eea7babae7f8cd89696e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b
    def get_inputs(self):
        return [
            paddle.uniform([9261, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2f56d38e40cb258115066d09088cf23a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([9261, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_831765e2b98b9a37fe2d11e9ecd4a5b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([4653, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4653, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f312430128df4434048240e6e5bad5a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1022, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1022, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_546d364a3be2c72efdd070564824f7a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_546d364a3be2c72efdd070564824f7a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_546d364a3be2c72efdd070564824f7a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_546d364a3be2c72efdd070564824f7a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_546d364a3be2c72efdd070564824f7a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_546d364a3be2c72efdd070564824f7a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_546d364a3be2c72efdd070564824f7a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_546d364a3be2c72efdd070564824f7a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_546d364a3be2c72efdd070564824f7a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_546d364a3be2c72efdd070564824f7a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_546d364a3be2c72efdd070564824f7a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_21a50b949a46359c33f1a60837b3d774(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b
    def get_inputs(self):
        return [
            paddle.uniform([2100, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_67dd1ecbca2884ee78c7cd1a7eea2215(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([2100, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f312430128df4434048240e6e5bad5a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([1022, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1022, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_515f52247eb0925eb57e78c260d58f96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_291a3baba83aebcfbb0959b81157d6f7
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
            paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_faa77adc4925f43471195fed3e1a9874(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.030757920816540718, 0.30942824482917786, 0.09276753664016724, 0.44252586364746094], [0.4536651074886322, 0.4457365870475769, 0.14740948379039764, 0.4085237383842468], [0.4536651074886322, 0.4457365870475769, 0.14740948379039764, 0.4085237383842468], [0.29568400979042053, 0.020557958632707596, 0.229517862200737, 0.2792818248271942], [0.02849540486931801, 0.052373725920915604, 0.2668256163597107, 0.037516992539167404], [0.44800955057144165, 0.4503563940525055, 0.4647688865661621, 0.37141361832618713]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([[0.04741102084517479, 0.12851227819919586, 0.42099282145500183, 0.06301568448543549], [0.06640581786632538, 0.46593543887138367, 0.36126700043678284, 0.2646188735961914], [0.06640581786632538, 0.46593543887138367, 0.36126700043678284, 0.2646188735961914], [0.12631738185882568, 0.42651644349098206, 0.20479850471019745, 0.09247107803821564], [0.17669986188411713, 0.0070557976141572, 0.24092066287994385, 0.1861073076725006], [0.3691565692424774, 0.4530242085456848, 0.4404996931552887, 0.09307760745286942]], dtype='float32').reshape([6, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_e954254e5fbefe9b4490b02c5f49f257(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_293d93bbcf6e674049780843ff641a48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([100, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([100, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a29c0039c443d7cc6c230e56d9f2c6b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([100, 1, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1.3999216556549072, 0.4594396948814392, 3.3227317333221436, 1.2254887819290161], [0.4531983733177185, 2.966006278991699, 1.7263309955596924, 58.01933670043945]], dtype='float32').reshape([2, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_1f89c3af7552a58fe9b15dcec9b41985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_07ed5983b01422c9e20bc8c43f0c8f5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19d0a8a77ea8ac0dd0fb5dcf3895371d
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5506a7d5add76bd5c93be0c3ec95da8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([300, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([300, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d7cf99d8bae10c3718889bd213ba63d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([300, 1, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[2.259181022644043, 0.43287745118141174, 1.924061894416809, 1.5772923231124878], [0.07533172518014908, 0.9091368317604065, 0.9618554711341858, 3.694631576538086]], dtype='float32').reshape([2, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7a90f3a423a71068378b1eb112b05497(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.19564540684223175], [0.034317974001169205], [0.29465341567993164], [0.24096974730491638], [0.33738017082214355]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.20026913285255432], [0.41332828998565674], [0.18956781923770905], [0.2602648437023163], [0.48564788699150085]], dtype='float32').reshape([5, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d1da23cbe8ea5a5e3cb742276145e27b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.034618474543094635], [0.3562788665294647], [0.11915521323680878], [0.15779650211334229], [0.047968797385692596]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.22313657402992249], [0.2731638550758362], [0.2620335519313812], [0.48791664838790894], [0.16293737292289734]], dtype='float32').reshape([5, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a8ebb5f145608899474b6b6a53eda95d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.19564540684223175], [0.034317974001169205], [0.29465341567993164], [0.24096974730491638], [0.33738017082214355]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.20026913285255432], [0.18348328769207], [0.18956781923770905], [0.2602648437023163], [0.2216607630252838]], dtype='float32').reshape([5, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_899829f0ae9fe5562eda5a3ea29cc1aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.12423448264598846], [0.3562788665294647], [0.11915521323680878], [0.15779650211334229], [0.047968797385692596]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.22313657402992249], [0.19915108382701874], [0.2620335519313812], [0.48791664838790894], [0.14740948379039764]], dtype='float32').reshape([5, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_647b3b41e294fcdcad24d8c77ca09182(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.37746039032936096], [0.16417308151721954], [0.38406670093536377], [0.27725130319595337], [0.4067143201828003]], dtype='float32').reshape([5, 1]),
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
class TestPrimitiveOp_662916f4af1fc905fcffd3634f4f3b50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.034618474543094635], [0.3611237704753876], [0.4964972734451294], [0.23624840378761292], [0.08820351958274841]], dtype='float32').reshape([5, 1]),
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
class TestPrimitiveOp_693aaa42f72c24eb5e6e62025a443624(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.00875263661146164], [-0.045353688299655914], [0.07074305415153503], [0.018018260598182678], [-0.0056082080118358135]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8c7305a0518e93f0151c6c562fadf8d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.37746039032936096], [0.16417308151721954], [0.38406670093536377], [0.27725130319595337], [0.4067143201828003]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.07452657073736191], [0.18348328769207], [0.12519223988056183], [0.003253175877034664], [0.2216607630252838]], dtype='float32').reshape([5, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_65fa390c7579f2dbc4260208736e8715(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.12423448264598846], [0.3611237704753876], [0.4964972734451294], [0.23624840378761292], [0.08820351958274841]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.00723513076081872], [0.19915108382701874], [0.16522662341594696], [0.19373512268066406], [0.14740948379039764]], dtype='float32').reshape([5, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_16488c14b697880480ab4e587ac7764b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.03544306010007858], [-0.0031277260277420282], [0.08575750887393951], [0.01164855994284153], [-0.010956274345517159]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.00875263661146164], [-0.045353688299655914], [0.07074305415153503], [0.018018260598182678], [-0.0056082080118358135]], dtype='float32').reshape([5, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_51b38bb19e03d14f9fb4f619552ab72f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [-0.0], [0.0], [0.0], [-0.0]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.7530507445335388], [-13.500530242919922], [0.175080344080925], [-0.5468230247497559], [0.4881281852722168]], dtype='float32').reshape([5, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9f31b45242dcc5467d9b1b85f0c9a8f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_291a3baba83aebcfbb0959b81157d6f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_32d88ff426f1ae89ee8cb29638b00110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_0a3f5e24b483c89e2cfe6cb2e1426590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_f2a92392dfc503f5f6c69915cce21eed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([2305, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2305, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_90822cad500d0cdea4c74fc5b8dcd557(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_90822cad500d0cdea4c74fc5b8dcd557(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_90822cad500d0cdea4c74fc5b8dcd557(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_90822cad500d0cdea4c74fc5b8dcd557(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_90822cad500d0cdea4c74fc5b8dcd557(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_90822cad500d0cdea4c74fc5b8dcd557(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_90822cad500d0cdea4c74fc5b8dcd557(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_90822cad500d0cdea4c74fc5b8dcd557(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_90822cad500d0cdea4c74fc5b8dcd557(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_90822cad500d0cdea4c74fc5b8dcd557(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_90822cad500d0cdea4c74fc5b8dcd557(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_0cf1b79d6b0bda99fe30897e159b4ad8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b
    def get_inputs(self):
        return [
            paddle.uniform([4725, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_482a57e3f8a9d1754788fd87985c8f0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([4725, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f2a92392dfc503f5f6c69915cce21eed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([2305, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2305, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_97fd90b7ae08446d8353003ac3cc71de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([3053, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3053, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dce40333703def1677a9508646dfcbcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_dce40333703def1677a9508646dfcbcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_dce40333703def1677a9508646dfcbcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_dce40333703def1677a9508646dfcbcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_dce40333703def1677a9508646dfcbcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_dce40333703def1677a9508646dfcbcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_dce40333703def1677a9508646dfcbcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_dce40333703def1677a9508646dfcbcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_dce40333703def1677a9508646dfcbcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_dce40333703def1677a9508646dfcbcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_dce40333703def1677a9508646dfcbcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_8d5f52ddf8383c305d82d62c38a7f453(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b
    def get_inputs(self):
        return [
            paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fc25a83b16b04b6ad5a3de78a8e76b5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_97fd90b7ae08446d8353003ac3cc71de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([3053, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3053, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_36bba7f0cc1832360cd418c5d5aed509(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([3712, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3712, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a8535d89b41d2e3a7368a1feb0ed9bba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_a8535d89b41d2e3a7368a1feb0ed9bba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_a8535d89b41d2e3a7368a1feb0ed9bba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_a8535d89b41d2e3a7368a1feb0ed9bba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_a8535d89b41d2e3a7368a1feb0ed9bba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_a8535d89b41d2e3a7368a1feb0ed9bba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_a8535d89b41d2e3a7368a1feb0ed9bba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_a8535d89b41d2e3a7368a1feb0ed9bba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_a8535d89b41d2e3a7368a1feb0ed9bba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_a8535d89b41d2e3a7368a1feb0ed9bba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_a8535d89b41d2e3a7368a1feb0ed9bba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_dba09ba03ffefda8f2726c71385338fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b
    def get_inputs(self):
        return [
            paddle.uniform([7581, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b9f8e1c85db4557fff30430af9b30c7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([7581, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_36bba7f0cc1832360cd418c5d5aed509(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([3712, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([3712, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_80c271ec998c36d91accf2075aa68a64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_291a3baba83aebcfbb0959b81157d6f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f3a508b496778ed0b643bc983d4abc57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_62d8fc1bb9de4a3d518136d70c332e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_e76620bcf0d595eda1ab5ed69cc6b979(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19d0a8a77ea8ac0dd0fb5dcf3895371d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b03ab04a7c6a80550adca44ff1953292(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([20]),
            paddle.to_tensor([0.4588078260421753, 0.068813256919384, 0.27381643652915955, 0.0062551796436309814, 0.2485503852367401, 0.32595863938331604, 0.3125576674938202, 0.4546583294868469, 0.21912570297718048, 0.04495468735694885, 0.2998899519443512, 0.22375978529453278, 0.08261338621377945, 0.3015618324279785, 0.13809457421302795, 0.3192104697227478, 0.31224656105041504, 0.3313934803009033, 0.4789752960205078, 0.3796774744987488], dtype='float32').reshape([20]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_44c3ff04e72ab58ad3e9d4458ca255c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_99bddb5fd63ecd93cc59d50704818428
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4588078260421753, 0.068813256919384, 0.27381643652915955, 0.0062551796436309814, 0.2485503852367401, 0.32595863938331604, 0.3125576674938202, 0.4546583294868469, 0.21912570297718048, 0.04495468735694885, 0.2998899519443512, 0.22375978529453278, 0.08261338621377945, 0.3015618324279785, 0.13809457421302795, 0.3192104697227478, 0.31224656105041504, 0.3313934803009033, 0.4789752960205078, 0.3796774744987488], dtype='float32').reshape([20]),
            paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([20]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ddf75c9f3c979bb9acc03625cbdf7fb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.03196848928928375], [0.24156440794467926], [0.0597226619720459], [0.32563695311546326]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.2075873762369156], [0.3073572814464569], [0.30402541160583496], [0.30721813440322876]], dtype='float32').reshape([4, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ef025180d9c70063a5865791baf4136f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06388451904058456], [0.14793279767036438], [0.12175875902175903], [0.26954853534698486]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.45928049087524414], [0.4224745035171509], [0.48921844363212585], [0.4449281692504883]], dtype='float32').reshape([4, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fdb1a6e2fd80014ccf6f505dd942966b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.11194626986980438], [0.3971475064754486], [0.26984596252441406], [0.32563695311546326]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.2075873762369156], [0.3073572814464569], [0.30402541160583496], [0.26391908526420593]], dtype='float32').reshape([4, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9cbed90ded4429f761c7fd1360f330ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06388451904058456], [0.2715681791305542], [0.12175875902175903], [0.30955228209495544]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.2821217179298401], [0.4224745035171509], [0.2759237587451935], [0.07581711560487747]], dtype='float32').reshape([4, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b09a7945f2d259abed9041b602fdaeb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.03196848928928375], [0.24156440794467926], [0.0597226619720459], [0.4864690601825714]], dtype='float32').reshape([4, 1]),
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
class TestPrimitiveOp_c310627930a62662838d5f5a20649c89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.17564170062541962], [0.14793279767036438], [0.27478042244911194], [0.26954853534698486]], dtype='float32').reshape([4, 1]),
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
class TestPrimitiveOp_2d9872f501403a66d0859da178ede8e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.034240152686834335], [-0.05106562748551369], [0.0040462748147547245], [-0.017011325806379318]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9bdff04d932a4faa51cdcafe984a839b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.11194626986980438], [0.3971475064754486], [0.26984596252441406], [0.4864690601825714]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.07909782230854034], [0.09549641609191895], [0.05401938408613205], [0.26391908526420593]], dtype='float32').reshape([4, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_54a438717cb3e1c8aad62a5de5b3bc7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.17564170062541962], [0.2715681791305542], [0.27478042244911194], [0.30955228209495544]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.2821217179298401], [0.4047701358795166], [0.2759237587451935], [0.07581711560487747]], dtype='float32').reshape([4, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d947b5bfd6646987f0ed9ea799d633d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.0034977032337337732], [-0.0401805154979229], [-0.0002467622689437121], [0.052017755806446075]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.034240152686834335], [-0.05106562748551369], [0.0040462748147547245], [-0.017011325806379318]], dtype='float32').reshape([4, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_60dd8d8dd0a6c736859961e2164b950f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [-0.0], [0.0], [-0.0]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[10.789324760437012], [-0.2709052264690399], [17.3974609375], [1.3270291090011597]], dtype='float32').reshape([4, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2bdf48ae6973d8a4487566beb1fae511(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_72c027a2256c56e42c71d76ad9cd17e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([2102, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2102, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b9317fd1a3c80d0b250fbefa313fe45d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_b9317fd1a3c80d0b250fbefa313fe45d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_b9317fd1a3c80d0b250fbefa313fe45d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_b9317fd1a3c80d0b250fbefa313fe45d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_b9317fd1a3c80d0b250fbefa313fe45d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_b9317fd1a3c80d0b250fbefa313fe45d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_b9317fd1a3c80d0b250fbefa313fe45d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_b9317fd1a3c80d0b250fbefa313fe45d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_b9317fd1a3c80d0b250fbefa313fe45d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_b9317fd1a3c80d0b250fbefa313fe45d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_b9317fd1a3c80d0b250fbefa313fe45d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_15aa17eea687877cf988c12a2c059c6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b
    def get_inputs(self):
        return [
            paddle.uniform([4116, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_358bc8c1eb98b6e964028dcc0f56dec2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([4116, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_72c027a2256c56e42c71d76ad9cd17e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([2102, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([2102, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e76620bcf0d595eda1ab5ed69cc6b979(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19d0a8a77ea8ac0dd0fb5dcf3895371d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a325bb76560ca99b51e435c001af07da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_291a3baba83aebcfbb0959b81157d6f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1505b6ac78a5252a396889880f9fe89b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19d0a8a77ea8ac0dd0fb5dcf3895371d
    def get_inputs(self):
        return [
            paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6a08ef289f3e60926465a198a8ca4acc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.09660603851079941, 0.3473486304283142, 0.1792222410440445, 0.18537181615829468], [0.041187599301338196, 0.4844166040420532, 0.3308883607387543, 0.04537099227309227], [0.3677883446216583, 0.28455665707588196, 0.04476252570748329, 0.1025569960474968], [0.3677883446216583, 0.28455665707588196, 0.04476252570748329, 0.1025569960474968], [0.173814058303833, 0.032673005014657974, 0.2887442708015442, 0.17421869933605194]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([[0.3407200574874878, 0.21765947341918945, 0.25243550539016724, 0.42701151967048645], [0.23007768392562866, 0.04049093276262283, 0.40273621678352356, 0.1927415430545807], [0.10648475587368011, 0.16337205469608307, 0.42770418524742126, 0.41350576281547546], [0.10648475587368011, 0.16337205469608307, 0.42770418524742126, 0.41350576281547546], [0.4299686849117279, 0.23296520113945007, 0.4033758342266083, 0.018267221748828888]], dtype='float32').reshape([5, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_04a7c952bd6d7d552aefd90150c9bae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_4dbb4969d19b1636651799c02c9cb347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_5eedd941575a8025b1f6142513fa4c9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_7318124dbfa7d00946c2412f1015cd99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([4228, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4228, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0bbbfb64b5dc2219a2b48b73ba235917(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_0bbbfb64b5dc2219a2b48b73ba235917(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_0bbbfb64b5dc2219a2b48b73ba235917(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_0bbbfb64b5dc2219a2b48b73ba235917(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_0bbbfb64b5dc2219a2b48b73ba235917(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_0bbbfb64b5dc2219a2b48b73ba235917(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_0bbbfb64b5dc2219a2b48b73ba235917(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_0bbbfb64b5dc2219a2b48b73ba235917(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_0bbbfb64b5dc2219a2b48b73ba235917(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_0bbbfb64b5dc2219a2b48b73ba235917(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_0bbbfb64b5dc2219a2b48b73ba235917(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
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
class TestPrimitiveOp_fadf83c28010ecd5bd8925902fc6620e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f482685ae5a5b9ab6841c9c12afa34b
    def get_inputs(self):
        return [
            paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0906bc96c54199c50c23766f6e035159(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e847dd8b2666f129b4077cb10bb59b7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7318124dbfa7d00946c2412f1015cd99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([4228, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([4228, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_46a383a87e08c137951d80a8dcedc762(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3381723165512085, 0.17756247520446777, 0.06380031257867813, 0.38561683893203735], [0.10080365091562271, 0.3514818549156189, 0.2887912094593048, 0.47205549478530884], [0.4350821375846863, 0.026866599917411804, 0.22386538982391357, 0.34845173358917236], [0.3381723165512085, 0.17756247520446777, 0.06380031257867813, 0.38561683893203735], [0.15402036905288696, 0.22495560348033905, 0.02383223921060562, 0.4998984634876251], [0.14966902136802673, 0.3276309669017792, 0.4284338355064392, 0.1937379390001297], [0.15402036905288696, 0.22495560348033905, 0.02383223921060562, 0.4998984634876251]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([[0.26299649477005005, 0.41595983505249023, 0.1868925839662552, 0.08783718943595886], [0.33425372838974, 0.35268688201904297, 0.04515431076288223, 0.4619998037815094], [0.30313247442245483, 0.06827733665704727, 0.1848611682653427, 0.08456997573375702], [0.26299649477005005, 0.41595983505249023, 0.1868925839662552, 0.08783718943595886], [0.3514348566532135, 0.43885645270347595, 0.2689119279384613, 0.44598138332366943], [0.4748520851135254, 0.467897891998291, 0.2615504264831543, 0.3895653486251831], [0.3514348566532135, 0.43885645270347595, 0.2689119279384613, 0.44598138332366943]], dtype='float32').reshape([7, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_ea7f130c75e933ee2ce60e9115ad4ea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2de6dca2a045b3dda212f3bb608192c8
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
class TestPrimitiveOp_5c1607c584dc7e104775c4f773344f40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b00cf09a76c3e09bb2e75ad6ee396540
    def get_inputs(self):
        return [
            paddle.uniform([52, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([52, 4], dtype='float32', min=0, max=0.5),
        ]


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