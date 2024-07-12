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
class PrimitiveOp_1848e60d49960e74b723e5f73421c99d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bb4ed6179b2a01843f7b0404f8c4d82d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 21504, 2], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_88bc0c57e668f9e2df7223998fb5ca57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 36], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_72c239abbbaca8cf2c19e6e35ade7786(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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


class PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cd8413f40117fe087071a90c85e9eb53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([300], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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


class PrimitiveOp_f106e9dda5232c93243dff7fc2676c69(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 18, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 9, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_44124b1631ce8b732a6280243ef6e81e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 24, 36], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_7d8e9bcd2df434a2ee27726613b97239(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_4c0d35f3b5a0f30e8f44d6221d391b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([8], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_7946f0ab3491edac3feb79686307bed7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.16903062164783478, 0.3801995515823364, 0.13765276968479156, 0.11802694201469421], [0.38483214378356934, 0.22154369950294495, 0.21937145292758942, 0.36948561668395996]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_97e60f0d8187517dbccc2e56bc869089(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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


class PrimitiveOp_1457086f8118f6ae44c94f8b23f05743(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c52b5fc607993875a61bf26722edbc5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1457086f8118f6ae44c94f8b23f05743
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_092a8d62914230a463eeb9d7215936c4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_edba7fda5fe10881888a3f3b2ba969ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 16, 16], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_144ad015232a4ec9babf8748384970a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_959ccfa6b5df262a29ac321871188c90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6e49408cda69bce22d03907e6a4ad98e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4d39cebbb9ce8b420f2268eb38ce6406(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_690b381e3831f63d2c4d59a651a79570(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([100], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_70bb88f93de413821c8609aac9786df4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 112, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 112, 160], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_9392b30c0e30f025725ed475fb5621b9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 80, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dc551b7c1c58b70a0dd7762fcf26dba3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 6, 6], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 6, 6], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_7c31944942426a78ec58610628e274a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 52, 52], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f711307187e6a459aa52e55d4d96254e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 64, 64], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_8d508cf2cbc394b8467027eb29c340e5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[96, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0510205a9c7efdd4d2de55a77f5418b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d508cf2cbc394b8467027eb29c340e5
    def get_inputs(self):
        return [
            paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_58d1fc007f768492e002012cdf5c79e9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[48, 48], dtype='float32'),
            paddle.static.InputSpec(shape=[48, 48], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a8ccc9c595004109166c3fae8d05cdc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58d1fc007f768492e002012cdf5c79e9
    def get_inputs(self):
        return [
            paddle.uniform([48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 48], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_0a4b9c90c22e6ec6569fdb099c53a9f1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
            paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1701b798f38005fd8e96c92e7e1e32af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a4b9c90c22e6ec6569fdb099c53a9f1
    def get_inputs(self):
        return [
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5c828b0cc0c36d336ec79f8430d27e58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0510205a9c7efdd4d2de55a77f5418b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d508cf2cbc394b8467027eb29c340e5
    def get_inputs(self):
        return [
            paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a8ccc9c595004109166c3fae8d05cdc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58d1fc007f768492e002012cdf5c79e9
    def get_inputs(self):
        return [
            paddle.uniform([48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 48], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1701b798f38005fd8e96c92e7e1e32af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a4b9c90c22e6ec6569fdb099c53a9f1
    def get_inputs(self):
        return [
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_353cef533498117db7a6e1f950fb36af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 60, 60], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8c6f2b983f9d859f3383c79a15bbeee4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_7b754fd89cd1714facabcab6a67ce3f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 11, 11], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 11, 11], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6ae8c2a25ecf2a08ee35979c9be99a9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.34958770871162415, 0.4914834797382355, 0.030791470780968666, 0.4862590730190277], [0.4058743715286255, 0.19737207889556885, 0.027092628180980682, 0.4146954119205475]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_97e60f0d8187517dbccc2e56bc869089(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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


class PrimitiveOp_536554d4068550f339721ff7091de2ab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_738ad3ab4bec68e89c6e3dbf4fc36676(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_536554d4068550f339721ff7091de2ab
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c1782ba106b739e0998af469f53e9819(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 30, 50], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 30, 50], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_747308f836dc0eb5fd5914713203fec5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 9, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 9, 9], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_b2129367263bf0a2cc9c6b900b6f92dc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9c9713b123b59730ec06ab5cdb0721f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2129367263bf0a2cc9c6b900b6f92dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.20786486566066742, 0.15852090716362]], [[0.3359413743019104, 0.38385361433029175]], [[0.10500519722700119, 0.088572658598423]], [[0.05509009584784508, 0.2576507329940796]], [[0.08082795888185501, 0.03169358894228935]], [[0.14791473746299744, 0.2231413871049881]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.36782288551330566, 0.23864783346652985]], [[0.32681968808174133, 0.39968106150627136]], [[0.12844915688037872, 0.11420886218547821]], [[0.1615632325410843, 0.4442180097103119]], [[0.2168838232755661, 0.020212797448039055]], [[0.12264090776443481, 0.462643563747406]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.18216867744922638, 0.01919618621468544]], [[0.22644925117492676, 0.1858130693435669]], [[0.4714430272579193, 0.4137498438358307]], [[0.07776288688182831, 0.44226565957069397]], [[0.41090327501296997, 0.1718730330467224]], [[0.06405521184206009, 0.37997767329216003]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.48425427079200745, 0.46140798926353455]], [[0.48020419478416443, 0.3810528516769409]], [[0.0325995534658432, 0.16867685317993164]], [[0.1398477405309677, 0.08869140595197678]], [[0.20558899641036987, 0.37943241000175476]], [[0.45008981227874756, 0.07396645098924637]]]], dtype='float32').reshape([1, 6, 1, 2]),
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


class PrimitiveOp_6a175d0c0530a3c5162117bbcd513875(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_63772588aea95c8fb6fb6491d67d62e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a175d0c0530a3c5162117bbcd513875
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.3828085660934448, 0.03431301191449165], [0.023540113121271133, 0.25350096821784973], [0.1122574508190155, 0.42175671458244324], [0.48164764046669006, 0.4113951027393341], [0.07737068086862564, 0.37415698170661926], [0.0703463926911354, 0.2796851694583893]]], dtype='float32').reshape([1, 6, 2]),
            paddle.to_tensor([[[0.2346169650554657, 0.24948328733444214], [0.2773289084434509, 0.49571195244789124], [0.20115098357200623, 0.48211953043937683], [0.4969460070133209, 0.4321368932723999], [0.07187119126319885, 0.46304598450660706], [0.45128872990608215, 0.42227795720100403]]], dtype='float32').reshape([1, 6, 2]),
            paddle.to_tensor([[[0.4205370247364044], [0.3870924711227417], [0.393877238035202], [0.1548210084438324], [0.3596498370170593], [0.4254036545753479]]], dtype='float32').reshape([1, 6, 1]),
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
class TestPrimitiveOp_07971b1ceb2a33e1f71089fbe82aab1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 7, 7], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d1b2370ecbb4efe49164afd11ec0b4bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 24, 24], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_7d4b4b7a886369b5abf5c9b7038c96f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9288c0d014de0f64db84d77370a363f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4417053759098053, 0.45320624113082886, 0.28990837931632996, 0.4012581706047058], [0.3686635196208954, 0.4158131778240204, 0.2523808479309082, 0.48548614978790283]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_97e60f0d8187517dbccc2e56bc869089(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_c6f323ffac33f8de847925ea46a7253e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_7d8e9bcd2df434a2ee27726613b97239(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_4c0d35f3b5a0f30e8f44d6221d391b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([8], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_dc93c5815c78619d759c2645eef014e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1457086f8118f6ae44c94f8b23f05743
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_5c3ae42c817ec934892ae13bad46f641(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300], dtype='float32'),
            paddle.static.InputSpec(shape=[300], dtype='float32'),
            paddle.static.InputSpec(shape=[300], dtype='float32'),
            paddle.static.InputSpec(shape=[300], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_adeaa095b592912bbd3a3f841506a5a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c3ae42c817ec934892ae13bad46f641
    def get_inputs(self):
        return [
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_94acec3b8c48b789729b93f51dc717e9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 40, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 40, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e5982e2aec6679593f25fdd4d8b48319(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94acec3b8c48b789729b93f51dc717e9
    def get_inputs(self):
        return [
            paddle.uniform([22, 40, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 40, 14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_979426c7c8aa6341f1f78e034608ae3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_12bf2ac8c534ef4d25941c2dc1369827(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 24, 24], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_cc65f589714792b7d937f9e9ec521abe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1457086f8118f6ae44c94f8b23f05743
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_07971b1ceb2a33e1f71089fbe82aab1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 7, 7], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c7d6a2ce7bf0245f9bed82faedd832d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 15, 15], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_58ff7aa29e7c894b62adecece32e7017(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_45753d63be77e5a306508a8b2d5e948f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 36, 36], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8c6f2b983f9d859f3383c79a15bbeee4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_69a3489015c06588a98698e2063d474b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_05687e69f3e46106f03d169c1590d4a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([1738, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1738, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1738, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1738, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_05687e69f3e46106f03d169c1590d4a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([1738, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1738, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1738, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1738, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_05b8bde67313b7d9c11fce33ea18321d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b4ab8938db859088e381c651a072219d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 8, 8], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_103c1f0030162dfbc30c343714668a20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1457086f8118f6ae44c94f8b23f05743
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_632825a5440eead2cdfc0d10e827dc22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([20, 30], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_6a94b94f403500d4ee1dde12643c597d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0d20f1964c4888a17821a1a626aea80e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a94b94f403500d4ee1dde12643c597d
    def get_inputs(self):
        return [
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_093122e485d209c87b4a878cf06c16f4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[32, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[32, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_88631d0e73f108117223f60cdea95a67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_093122e485d209c87b4a878cf06c16f4
    def get_inputs(self):
        return [
            paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_b63825d1a4ae907c88ba57f8f5b33c40(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[16, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_25051e3f9454a43e201351daa822d29c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b63825d1a4ae907c88ba57f8f5b33c40
    def get_inputs(self):
        return [
            paddle.uniform([16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 16], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_068144b324fbc9d974dcb67738c5cad6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0d20f1964c4888a17821a1a626aea80e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a94b94f403500d4ee1dde12643c597d
    def get_inputs(self):
        return [
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_88631d0e73f108117223f60cdea95a67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_093122e485d209c87b4a878cf06c16f4
    def get_inputs(self):
        return [
            paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_25051e3f9454a43e201351daa822d29c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b63825d1a4ae907c88ba57f8f5b33c40
    def get_inputs(self):
        return [
            paddle.uniform([16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 16], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f4b3197f508a111fde8b160896295bdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.11145345121622086, 0.23862624168395996, 0.405653178691864, 0.05646820738911629], [0.4446398913860321, 0.4059602618217468, 0.3826504349708557, 0.2837061583995819], [0.06379975378513336, 0.22761933505535126, 0.051665134727954865, 0.4780203104019165], [0.1788037121295929, 0.17385177314281464, 0.37165603041648865, 0.09100950509309769], [0.21199457347393036, 0.07097514718770981, 0.23687568306922913, 0.35021036863327026], [0.029423732310533524, 0.006402377039194107, 0.20529043674468994, 0.2994297742843628], [0.13569502532482147, 0.2893596589565277, 0.11089305579662323, 0.09220270812511444]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_5f2a9a34d9ce095a6d373a978df49d93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([7], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_4f0f87da5d1e5a66cd8e407e42a08944(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 48, 48], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_0bf89928138f5b24110587c67e8fa6e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 48, 72], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c52b5fc607993875a61bf26722edbc5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1457086f8118f6ae44c94f8b23f05743
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_34b5bfe351392a61a4919e8b0fb9b096(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dda2a47081d8eb070b1881dc1fd6993a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34b5bfe351392a61a4919e8b0fb9b096
    def get_inputs(self):
        return [
            paddle.uniform([22, 28, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 28, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 28, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 28, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 28, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 28, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 28, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 28, 28, 28], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_e734ca0b10721d7d688a261329cb981a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_46504505bc8b51184e0493f57585a41d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3228645920753479], [0.3960472047328949], [0.3066532611846924], [0.021218301728367805], [0.2319512814283371], [0.3718463182449341], [0.08288423717021942], [0.33881354331970215], [0.38703471422195435]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.22029320895671844], [0.11446359008550644], [0.09660761803388596], [0.22756057977676392], [0.20094726979732513], [0.00021650124108418822], [0.06975062936544418], [0.1015438437461853], [0.14690779149532318]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.3899255692958832], [0.01601838320493698], [0.2937104105949402], [0.18330015242099762], [0.02360283024609089], [0.38878753781318665], [0.2861289083957672], [0.20783261954784393], [0.16109074652194977]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.0589759387075901], [0.4124438464641571], [0.06439685821533203], [0.264473021030426], [0.11398261785507202], [0.15532927215099335], [0.11311130225658417], [0.4885890781879425], [0.38761046528816223]], dtype='float32').reshape([9, 1]),
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
class TestPrimitiveOp_3417fe1fd707f662f34efad197e2446d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4389498233795166], [0.15551380813121796], [0.15075920522212982], [0.32667702436447144], [0.1646711528301239], [0.46801039576530457], [0.43087238073349], [0.4495995342731476], [0.2844972312450409]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.01866067200899124], [0.43595606088638306], [0.27865275740623474], [0.24296759068965912], [0.07518929243087769], [0.087107814848423], [0.4223915636539459], [0.44737523794174194], [0.3034294545650482]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.018780270591378212], [0.3621695339679718], [0.48838600516319275], [0.15385977923870087], [0.4510036110877991], [0.35555529594421387], [0.32748493552207947], [0.21274198591709137], [0.13382747769355774]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.19809138774871826], [0.1797289103269577], [0.031369950622320175], [0.1931517869234085], [0.29963502287864685], [0.2445525825023651], [0.03370177373290062], [0.06182239204645157], [0.3453497588634491]], dtype='float32').reshape([9, 1]),
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
class TestPrimitiveOp_3f92a555e56ec4598276ce514961795b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 60, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 60, 100], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6933fc52eaa223c14c2741bab6c14bb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a584fcd652d89ea075bfce9f2062e1b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f395f03fafee203c555a94643e375026(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f6d6a9ddb8f95fbf71f1c24be8d10219(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 16, 32], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9702dd4de525a1a0e32240c2d0e4d5a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 4, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_352d630cf196ea3ad4dc1feb886ac2f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([5553, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5553, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5553, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5553, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_352d630cf196ea3ad4dc1feb886ac2f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([5553, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5553, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5553, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5553, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b5f7f362f5ad5b569ecb7d82e6494e97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_fd63e9365b1931d59162ae47dc77e0b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_08a20ddc171367f6a0d680a9034aab80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.024022752419114113, 0.38020074367523193, 0.34132951498031616, 0.44830092787742615], [0.23893548548221588, 0.15022563934326172, 0.3468237817287445, 0.3618110716342926], [0.41037508845329285, 0.2736871540546417, 0.4043949544429779, 0.17460180819034576], [0.2557991147041321, 0.2483685314655304, 0.19695279002189636, 0.13544489443302155], [0.08377285301685333, 0.4270392656326294, 0.4616359770298004, 0.27932026982307434], [0.4875655174255371, 0.26046931743621826, 0.03341309726238251, 0.3536261022090912]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_3af90ab907c9af06442c56824c7738ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([6], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_3f92a555e56ec4598276ce514961795b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 60, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 60, 100], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_7f6b8896b417264f9f0afe8e219cefe9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 28, 28], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_eeb0b9bab044b09395736d2a9f87a00e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 8, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f124bab359a957b381a6b58944d87fde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eeb0b9bab044b09395736d2a9f87a00e
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 8, 112, 112], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_521b3532a5cf474d80a43a789797cc1d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5c160a65581f855c57038ac0613bde1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_521b3532a5cf474d80a43a789797cc1d
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c439ac22172ead5b360fe310e403ed72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2717084288597107, 0.45454221963882446, 0.006341543979942799, 0.13451623916625977], [0.47601041197776794, 0.38216322660446167, 0.012220052070915699, 0.04670698568224907], [0.3812931478023529, 0.37026315927505493, 0.3200012743473053, 0.1258433759212494]], dtype='float32').reshape([3, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_ddc5f29dbd04067f08de4222dfec09e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([3], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_4a1124fc591ac9867819bf737c738110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 34, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 34, 34], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_119b12e7f83ba8739a2ea9c81b0d1603(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 7, 10], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_24d21c474b6d10b419d2aa8de3ff16f1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 120, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 120, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7fcaf03c4a02d1b0253f5797aedc251f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24d21c474b6d10b419d2aa8de3ff16f1
    def get_inputs(self):
        return [
            paddle.uniform([22, 120, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 120, 28, 28], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_372b53cca9711d495286672dc7f1828b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 44, 44], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f0740cf76dfcc5c2cbf7abd4fdc48a1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34b5bfe351392a61a4919e8b0fb9b096
    def get_inputs(self):
        return [
            paddle.uniform([22, 28, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 28, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 28, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 28, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 28, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 28, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 28, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 28, 56, 56], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d4bb432e730ad8ef66231d92c5203859(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 18, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 18, 18], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_499fbde81d3808b6ea0a46a48f40c6f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 17], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 17, 17], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_37cf37afe65d5db7a40d6a8556820c90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24d21c474b6d10b419d2aa8de3ff16f1
    def get_inputs(self):
        return [
            paddle.uniform([22, 120, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 120, 14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9e1538baf0702a5a965023a2fbe99923(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34b5bfe351392a61a4919e8b0fb9b096
    def get_inputs(self):
        return [
            paddle.uniform([10, 28, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 28, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 28, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 28, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 28, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 28, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 28, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 28, 56, 56], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_cac049235f5ed539d2fc14f4aa46d3e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 22, 22], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_d5dc53de0d2688a5b4748447c9855df4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_14792fffd2c65a7dd2ffa3464aa72e3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5dc53de0d2688a5b4748447c9855df4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.48949873447418213, 0.2125146985054016, 0.3943987786769867, 0.052135169506073, 0.3060843348503113, 0.14483611285686493], dtype='float32').reshape([6]),
            paddle.to_tensor([0.1455189287662506, 0.18382209539413452, 0.022990306839346886, 0.4498473107814789, 0.47428104281425476, 0.17863892018795013], dtype='float32').reshape([6]),
            paddle.to_tensor([0.033982645720243454, 0.30670925974845886, 0.2704288363456726, 0.40632981061935425, 0.399911105632782, 0.4827656149864197], dtype='float32').reshape([6]),
            paddle.to_tensor([0.46267789602279663, 0.06502489745616913, 0.23793815076351166, 0.4408537447452545, 0.28138476610183716, 0.4201972782611847], dtype='float32').reshape([6]),
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
class TestPrimitiveOp_3d7a6da84e68db53fff9a848dc937ca6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5dc53de0d2688a5b4748447c9855df4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3529529273509979, 0.1577829271554947, 0.3188677430152893, 0.33222490549087524, 0.2642192542552948, 0.17672815918922424], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2904326915740967, 0.35068219900131226, 0.13571391999721527, 0.24345217645168304, 0.447722464799881, 0.3805902302265167], dtype='float32').reshape([6]),
            paddle.to_tensor([0.4372062087059021, 0.4139530062675476, 0.03780849650502205, 0.035732731223106384, 0.03976374492049217, 0.21069513261318207], dtype='float32').reshape([6]),
            paddle.to_tensor([0.28935226798057556, 0.455740362405777, 0.22648458182811737, 0.2802315354347229, 0.46797147393226624, 0.46781590580940247], dtype='float32').reshape([6]),
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
class TestPrimitiveOp_8e8e0bfdefd0d0940c778247d0e774dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([1733, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1733, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1733, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1733, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8e8e0bfdefd0d0940c778247d0e774dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([1733, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1733, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1733, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1733, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_05b8bde67313b7d9c11fce33ea18321d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a5853ed03cf16091682292f1bb647ce3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 5, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 5, 5], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_de007bd9a6ceecee7266f8c81ecb4d7c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e933cadd3d66798db273d8805dd117f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de007bd9a6ceecee7266f8c81ecb4d7c
    def get_inputs(self):
        return [
            paddle.uniform([4, 144, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 144, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 144, 768], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8c6f2b983f9d859f3383c79a15bbeee4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6933fc52eaa223c14c2741bab6c14bb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a584fcd652d89ea075bfce9f2062e1b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f395f03fafee203c555a94643e375026(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c6f323ffac33f8de847925ea46a7253e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_cd2b1559c3a27fb42768e7d5929e4eca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.08432912081480026, 0.4075654149055481, 0.42063939571380615, 0.2695857286453247], [0.06464184820652008, 0.2741512060165405, 0.11042358726263046, 0.23393476009368896]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_97e60f0d8187517dbccc2e56bc869089(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_b2f66766bb20ccfa8fb08852128e9249(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1457086f8118f6ae44c94f8b23f05743
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6a7f11b32046650a1d478769d7b573d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.30835163593292236, 0.07124757021665573, 0.18481184542179108, 0.10005176812410355]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_c451a4630aa40476f8c12a2dd8b667f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_a00e76e3ff79a2bd5e025d26b6a16aa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 17, 128, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_64a159158ae7661df5ba7e3ad51dc217(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 26, 26], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_67bd6782fa2a543cd39bf781531d17db(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[80, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[80, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fae74374c8409a262a23a3b7cc999dd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67bd6782fa2a543cd39bf781531d17db
    def get_inputs(self):
        return [
            paddle.uniform([80, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 80], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_15ec87db4e5f15620d1e1a3ba7cb20cb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[40, 40], dtype='float32'),
            paddle.static.InputSpec(shape=[40, 40], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5a1259edcf78f64e65cb4acab5d68ec5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15ec87db4e5f15620d1e1a3ba7cb20cb
    def get_inputs(self):
        return [
            paddle.uniform([40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([40, 40], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_0a80ba214ef30b26b24f63ad9db0f30c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20, 20], dtype='float32'),
            paddle.static.InputSpec(shape=[20, 20], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_68971d2d93a35cc4a967696a965e7b3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a80ba214ef30b26b24f63ad9db0f30c
    def get_inputs(self):
        return [
            paddle.uniform([20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([20, 20], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_70b9d3bbbd752904d5da78c302fc4f5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fae74374c8409a262a23a3b7cc999dd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67bd6782fa2a543cd39bf781531d17db
    def get_inputs(self):
        return [
            paddle.uniform([80, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 80], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5a1259edcf78f64e65cb4acab5d68ec5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15ec87db4e5f15620d1e1a3ba7cb20cb
    def get_inputs(self):
        return [
            paddle.uniform([40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([40, 40], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_68971d2d93a35cc4a967696a965e7b3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a80ba214ef30b26b24f63ad9db0f30c
    def get_inputs(self):
        return [
            paddle.uniform([20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([20, 20], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_475f1705db145005d19fd6cad5a379b3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 49, 8, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 49, 8, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 49, 8, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e14024133ec17691379f3028a0523d7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_475f1705db145005d19fd6cad5a379b3
    def get_inputs(self):
        return [
            paddle.uniform([22, 49, 8, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 49, 8, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 49, 8, 32], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_7f0363af6fdb977e9199522a6ccfaa97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 13, 13], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_b2a7fb5cf0c88681876d3b0e6adc45f4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 160, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 160, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 160, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c305886d3a6f8da17ef9467f6ff7c6fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2a7fb5cf0c88681876d3b0e6adc45f4
    def get_inputs(self):
        return [
            paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6a1b3a1c6e2bc037d94413a40c69bf45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 36, 36], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1dcedba4fb0f6e11120f19dc6dca36ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a88937a4ba4b9baec3388c0967dd94bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34b5bfe351392a61a4919e8b0fb9b096
    def get_inputs(self):
        return [
            paddle.uniform([22, 112, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 112, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 112, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 112, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 112, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 112, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 112, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 112, 14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_cc54c9821270174de1a21835190105e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4f0f87da5d1e5a66cd8e407e42a08944(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 48, 48], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_144ad015232a4ec9babf8748384970a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_959ccfa6b5df262a29ac321871188c90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6e49408cda69bce22d03907e6a4ad98e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_69ef002098dbedb69e95fad437de154f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1457086f8118f6ae44c94f8b23f05743
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c59e8d72179d6b8507ce933f4fb0b1b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([1466, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1466, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1466, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1466, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c59e8d72179d6b8507ce933f4fb0b1b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([1466, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1466, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1466, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1466, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4233c14a8059d68bd80d489a3d3ae3a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6c0cc4c22ad720106d9d330841ac38a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 30, 30], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_77bc556f36c2162ce7909f25b18d7ccb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34b5bfe351392a61a4919e8b0fb9b096
    def get_inputs(self):
        return [
            paddle.uniform([22, 56, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 56, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 56, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 56, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 56, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 56, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 56, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 56, 28, 28], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_2f48e5b5d6732576fecf1a149071e411(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.21787340939044952, 0.3058196008205414, 0.1441837102174759, 0.4488373398780823], [0.41595470905303955, 0.12608909606933594, 0.16094008088111877, 0.10529551655054092], [0.17940425872802734, 0.05010313168168068, 0.2323574274778366, 0.29729440808296204], [0.2608910799026489, 0.2676137089729309, 0.11929520219564438, 0.0292961522936821], [0.17230208218097687, 0.19723495841026306, 0.4367802143096924, 0.26049256324768066], [0.2738981544971466, 0.11974108219146729, 0.27832460403442383, 0.24150776863098145], [0.21795953810214996, 0.30982330441474915, 0.22795483469963074, 0.3849623203277588]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_5f2a9a34d9ce095a6d373a978df49d93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([7], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_a466c1dbbc90d09b0e38455bbfa325b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.14669905602931976, 0.11785608530044556, 0.31231316924095154, 0.007499780505895615]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_c451a4630aa40476f8c12a2dd8b667f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_5047874ce14665dd3653812dce54ca6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34b5bfe351392a61a4919e8b0fb9b096
    def get_inputs(self):
        return [
            paddle.uniform([10, 14, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 14, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 14, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 14, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 14, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 14, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 14, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 14, 56, 56], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_a8c97900aa90cbd9ed273fb452623471(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 80, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 80, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 80, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f51834900e58625088829918fdebb71c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8c97900aa90cbd9ed273fb452623471
    def get_inputs(self):
        return [
            paddle.uniform([22, 80, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 80, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 80, 28, 28], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_cc54c9821270174de1a21835190105e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_57d62f97ab2dafacdcba7dcbd74f882e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.07675330340862274]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.04051472991704941]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.2655336260795593]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.06180190667510033]], dtype='float32').reshape([1, 1]),
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
class TestPrimitiveOp_5bc3f85f9c7d55ae9455c85656deef2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.399107426404953]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.11604505032300949]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.2398943156003952]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.368866503238678]], dtype='float32').reshape([1, 1]),
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


class PrimitiveOp_71067eebaf355996c10a9d4e28f761ca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 196, 8, None], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 196, 8, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_36c6ebb7c0a3c180ff8d91f173024381(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71067eebaf355996c10a9d4e28f761ca
    def get_inputs(self):
        return [
            paddle.uniform([22, 196, 8, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 196, 8, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b855b7d6743cf85f5daf27b6671d4453(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4892083704471588], [0.27091413736343384], [0.17127905786037445], [0.1577376276254654], [0.41942358016967773], [0.17888130247592926]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.21311093866825104], [0.05007264018058777], [0.38398033380508423], [0.09370165318250656], [0.32851335406303406], [0.1493993103504181]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.3479418456554413], [0.34206390380859375], [0.08347898721694946], [0.061992086470127106], [0.2510312795639038], [0.4961717128753662]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.09062915295362473], [0.18488666415214539], [0.34276267886161804], [0.2707947790622711], [0.3789082169532776], [0.28018254041671753]], dtype='float32').reshape([6, 1]),
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
class TestPrimitiveOp_646cb91487bfc6a54ae6c39fba8468f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06633955240249634], [0.18247836828231812], [0.2487204223871231], [0.44636279344558716], [0.2324966937303543], [0.3986716568470001]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.37008339166641235], [0.3361941874027252], [0.14341436326503754], [0.14076094329357147], [0.3007638156414032], [0.4232746660709381]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.49149706959724426], [0.3978572487831116], [0.2893059551715851], [0.04706785827875137], [0.09867657721042633], [0.3818233013153076]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.07251711189746857], [0.2581758201122284], [0.3393520414829254], [0.4828045070171356], [0.3792950510978699], [0.3344321846961975]], dtype='float32').reshape([6, 1]),
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


class PrimitiveOp_8757ff1dd8a466f4187c9c52f922399a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 300, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 300, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 300, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 300, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_136526e4ae24988a63676be828cf4c52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8757ff1dd8a466f4187c9c52f922399a
    def get_inputs(self):
        return [
            paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_dc551b7c1c58b70a0dd7762fcf26dba3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 6, 6], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 6, 6], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_8b04bb0e505aa517ac1b97d645951e0b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b64fa05dc9ef1967fa01e4f1472f7157(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b04bb0e505aa517ac1b97d645951e0b
    def get_inputs(self):
        return [
            paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_514e2b7832361ee7ba5c03c69f8f57fa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0b5fa6b219c737bfe92a8959930f84ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_514e2b7832361ee7ba5c03c69f8f57fa
    def get_inputs(self):
        return [
            paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_21554e657e43f93a58b530ef9ae88388(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3a34f249e62c4bb3fadf15c55c13b01e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21554e657e43f93a58b530ef9ae88388
    def get_inputs(self):
        return [
            paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_e6b1002cf5bd85949783ef2c25a960e4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7541237d00e1f42ae5579df35ad46461(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6b1002cf5bd85949783ef2c25a960e4
    def get_inputs(self):
        return [
            paddle.uniform([4, 576, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 576, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 576, 384], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_16f467c540e90b8d1b2b841df1e46fde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 32, 32], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f8b9f39a505a53e766fedda6b87d76cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.044476039707660675, 0.05122298002243042, 0.22665901482105255, 0.20944929122924805], [0.27527180314064026, 0.1378462016582489, 0.24943381547927856, 0.14275483787059784], [0.1291610151529312, 0.4926948547363281, 0.3651009202003479, 0.2669443190097809], [0.38224121928215027, 0.27661895751953125, 0.4803200960159302, 0.21076464653015137], [0.3033926486968994, 0.44697821140289307, 0.4121702313423157, 0.30973970890045166]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_8d33ee0f7804e19ae5fa954d05146406(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([5], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_abd77732e1cda0e599374a93b172b687(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.153085395693779, 0.15082640945911407, 0.3437725007534027, 0.08953218162059784], [0.22880303859710693, 0.3738715946674347, 0.27926191687583923, 0.27835729718208313], [0.14348462224006653, 0.010167037136852741, 0.28419172763824463, 0.32861584424972534], [0.24178385734558105, 0.2601092755794525, 0.32565972208976746, 0.07134416699409485], [0.16305334866046906, 0.3240160644054413, 0.32681068778038025, 0.030790362507104874], [0.3719404935836792, 0.41742363572120667, 0.3284975290298462, 0.010256124660372734], [0.30644646286964417, 0.2166462540626526, 0.11804600059986115, 0.16275270283222198]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_5f2a9a34d9ce095a6d373a978df49d93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([7], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_cc54c9821270174de1a21835190105e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_2902d48a8a9c1f84488a57c7d80c6773(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.17277765274047852, 0.4062096178531647, 0.13662110269069672, 0.13183529675006866], [0.1836676299571991, 0.33742737770080566, 0.0845433846116066, 0.39738979935646057], [0.18749657273292542, 0.4540046155452728, 0.3578137457370758, 0.2902390658855438], [0.29468220472335815, 0.2598723769187927, 0.4815906584262848, 0.4595033824443817], [0.3465569019317627, 0.46954914927482605, 0.09371209889650345, 0.08958199620246887], [0.17359988391399384, 0.4599875211715698, 0.41239801049232483, 0.13032981753349304], [0.3529101610183716, 0.3771044611930847, 0.2441311925649643, 0.3322638273239136]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_5f2a9a34d9ce095a6d373a978df49d93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([7], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_0e04b5c38fcf5d2371528b137bc751e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 96, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 96, 96], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_2f0cb283597fffeb5d2ffb37286cedd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 88, 88], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 88, 88], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_789e9529cf4e8ce6f2591f88f8679d8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.22347691655158997, 0.21912971138954163, 0.026322772726416588, 0.28863784670829773]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_c451a4630aa40476f8c12a2dd8b667f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_72c239abbbaca8cf2c19e6e35ade7786(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_cd8413f40117fe087071a90c85e9eb53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([300], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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


class PrimitiveOp_8681c1a1016d2414d6f73dfd91b7dc98(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7877e551beef61234d148b30b57aefa1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8681c1a1016d2414d6f73dfd91b7dc98
    def get_inputs(self):
        return [
            paddle.uniform([4, 2304, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 2304, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 2304, 192], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f3ca06e9d4e684c2f9d147a77d4cb5c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 80, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 80, 80], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_2b8357568a9745f62dc6f9f04b0c92dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([2052, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2052, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2052, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2052, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_2b8357568a9745f62dc6f9f04b0c92dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([2052, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2052, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2052, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2052, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_31f3771d77b3eb287bde90ee8b122515(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_5b94c1e4afa16f5d606b25312e3a9502(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 90, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 90, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 90, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 90, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ff089a0cf0b96f549c98387124297839(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b94c1e4afa16f5d606b25312e3a9502
    def get_inputs(self):
        return [
            paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1701b798f38005fd8e96c92e7e1e32af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a4b9c90c22e6ec6569fdb099c53a9f1
    def get_inputs(self):
        return [
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_bea112607e2311e80da5e571a9c86161(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 40, 40], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_17d05eb6382d2ea58862dbdb8e9865f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34b5bfe351392a61a4919e8b0fb9b096
    def get_inputs(self):
        return [
            paddle.uniform([22, 14, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 14, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 14, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 14, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 14, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 14, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 14, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 14, 56, 56], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_84833a8c576c88ad654afd914b4c47a9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2edb5833cb02b40945ae785e965f66e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84833a8c576c88ad654afd914b4c47a9
    def get_inputs(self):
        return [
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d1b2370ecbb4efe49164afd11ec0b4bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 24, 24], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_228df5ed2f0f055b448f0884a75f934f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([4717, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4717, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4717, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4717, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_228df5ed2f0f055b448f0884a75f934f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([4717, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4717, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4717, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4717, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_78bad22fe49e21cec6a67e7126a6ddd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9694e6ff00cb6491a017396a14f18581(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 160, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 17, 160, 160], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_bf825eb8319922fafc68039371329d21(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 600, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 600, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_74bf508f92c72f41168fb4bfb8cc4118(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf825eb8319922fafc68039371329d21
    def get_inputs(self):
        return [
            paddle.uniform([22, 600, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 600, 7, 7], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1027def907793453c5e70541607bfb19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.019870752468705177, 0.34105750918388367, 0.08597230911254883, 0.22628436982631683], [0.06770025938749313, 0.42492666840553284, 0.3536209166049957, 0.010973446071147919], [0.47830554842948914, 0.4089871644973755, 0.4721699357032776, 0.07909690588712692], [0.12308020889759064, 0.19180642068386078, 0.40730535984039307, 0.2070724219083786], [0.33429089188575745, 0.011158177629113197, 0.08848989009857178, 0.46986207365989685]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_8d33ee0f7804e19ae5fa954d05146406(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([5], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_0bf89928138f5b24110587c67e8fa6e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 48, 72], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f460f4d25466eea9925e1876b561bc04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([1056, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1056, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1056, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1056, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f460f4d25466eea9925e1876b561bc04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([1056, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1056, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1056, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1056, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c63d5f01dedb07a889a46f7cc76af9ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c6f323ffac33f8de847925ea46a7253e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f4e6ed5bdb7d782bc788cf9d057035bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 32, 32], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b693a8b44c9655702db6377742531e94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 15, 25], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 15, 25], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f9a9c6107356ebb24be6f20dab5ae6cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1457086f8118f6ae44c94f8b23f05743
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9ac88113545a0a3ec925f82b1167737d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 96, 144], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b5d5b2483abbdf6f0c72379f92925368(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 192, 288], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ca132ebf48f034a1a571349add59d0e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34b5bfe351392a61a4919e8b0fb9b096
    def get_inputs(self):
        return [
            paddle.uniform([22, 56, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 56, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 56, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 56, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 56, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 56, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 56, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 56, 14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_e061998d06205e4f99bd1202bd6bb114(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de007bd9a6ceecee7266f8c81ecb4d7c
    def get_inputs(self):
        return [
            paddle.uniform([6, 144, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 144, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 144, 768], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_f8e2dc07edfa99a53f8b4224f798d8ab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[68, 68], dtype='float32'),
            paddle.static.InputSpec(shape=[68, 68], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3dde0d1425551a91ebae0bfc1a0d75bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8e2dc07edfa99a53f8b4224f798d8ab
    def get_inputs(self):
        return [
            paddle.uniform([68, 68], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 68], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_ccead173ab5dc2a7a8b05130fcaeff4f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[34, 34], dtype='float32'),
            paddle.static.InputSpec(shape=[34, 34], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_86690e4a23c10710502f19ad4e2f19e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccead173ab5dc2a7a8b05130fcaeff4f
    def get_inputs(self):
        return [
            paddle.uniform([34, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([34, 34], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_1febda3a454828d94ffc748cf89b0a08(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[17, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[17, 17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ef330fbb9b0bd025328a7ebcf9019ece(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1febda3a454828d94ffc748cf89b0a08
    def get_inputs(self):
        return [
            paddle.uniform([17, 17], dtype='float32', min=0, max=0.5),
            paddle.uniform([17, 17], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d5c47e276a7632be0d355052650efdf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3dde0d1425551a91ebae0bfc1a0d75bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8e2dc07edfa99a53f8b4224f798d8ab
    def get_inputs(self):
        return [
            paddle.uniform([68, 68], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 68], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_86690e4a23c10710502f19ad4e2f19e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccead173ab5dc2a7a8b05130fcaeff4f
    def get_inputs(self):
        return [
            paddle.uniform([34, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([34, 34], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ef330fbb9b0bd025328a7ebcf9019ece(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1febda3a454828d94ffc748cf89b0a08
    def get_inputs(self):
        return [
            paddle.uniform([17, 17], dtype='float32', min=0, max=0.5),
            paddle.uniform([17, 17], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_fa33e9edb6d923d6a9aafbf9c05001e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.121558278799057], [0.026957757771015167], [0.3154665231704712], [0.13702236115932465], [0.13035985827445984]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.06055105850100517], [0.3023223578929901], [0.29442358016967773], [0.12275972962379456], [0.3305083215236664]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.23420564830303192], [0.1628895252943039], [0.17040301859378815], [0.04161243140697479], [0.31444212794303894]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.12737542390823364], [0.2753457725048065], [0.42593345046043396], [0.11517052352428436], [0.06334847956895828]], dtype='float32').reshape([5, 1]),
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
class TestPrimitiveOp_5002644b2a522fec502c5c3840d4ddd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.29517462849617004], [0.09762095659971237], [0.39388298988342285], [0.45222628116607666], [0.37068048119544983]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.015375080518424511], [0.07627338171005249], [0.2160526067018509], [0.0893891453742981], [0.40267425775527954]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.20128846168518066], [0.1977842003107071], [0.1439625471830368], [0.07181569188833237], [0.2501826286315918]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.2253778576850891], [0.03415966406464577], [0.43615180253982544], [0.47151103615760803], [0.14356163144111633]], dtype='float32').reshape([5, 1]),
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
class TestPrimitiveOp_4fe36439af7752ecd58faafc124f0a52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.28491389751434326, 0.25443458557128906, 0.024227358400821686, 0.31417572498321533]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_c451a4630aa40476f8c12a2dd8b667f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_8ee462787e16133403b927ba886fa509(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 68, 68], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 68, 68], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9ac88113545a0a3ec925f82b1167737d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 96, 144], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6e49408cda69bce22d03907e6a4ad98e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_959ccfa6b5df262a29ac321871188c90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_144ad015232a4ec9babf8748384970a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f60c81c970ad1c48f24c5a7d5db3dd2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 16], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_379b64667ba43ad8a6303d8f433aa279(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([8, 8], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_63cb58fa23a3f6d184ec8519e1128f95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 72, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 72, 72], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_ef8c2878092d3cbb853a14c2d58ae1d9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 36, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 36, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9d682f7195787490ff249f1494355d0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef8c2878092d3cbb853a14c2d58ae1d9
    def get_inputs(self):
        return [
            paddle.uniform([22, 36, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 36, 56, 56], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_7140e540c850cc1ae5872af22c014961(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1457086f8118f6ae44c94f8b23f05743
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_60649a0a13347a8bfb315367c178d719(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e4f42f414a63e386f3d1dda92eda8838(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60649a0a13347a8bfb315367c178d719
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 9216, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 9216, 96], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_715d31ab33200caceb33d78602eb679e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1fa19b01497b9e068fdcaba1606d451c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.10112754255533218, 0.10138582438230515, 0.4006735682487488, 0.376541405916214]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_c451a4630aa40476f8c12a2dd8b667f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_4870f1f9e7f83a33bff195cb6b8129b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([15, 25], dtype='float32', min=0, max=0.5),
            paddle.uniform([15, 25], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_14fc5f240e2cb1646d1c5230c304e9b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34b5bfe351392a61a4919e8b0fb9b096
    def get_inputs(self):
        return [
            paddle.uniform([22, 112, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 112, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 112, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 112, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 112, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 112, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 112, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 112, 7, 7], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_0a63ab489373738309c7ef609fd7f084(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([2354, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2354, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2354, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2354, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_0a63ab489373738309c7ef609fd7f084(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([2354, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2354, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2354, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2354, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b608b7e54c3e9c0db3bd2ae1327a0621(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1dcedba4fb0f6e11120f19dc6dca36ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_7d8e9bcd2df434a2ee27726613b97239(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_4c0d35f3b5a0f30e8f44d6221d391b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([8], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_4fa2002e6d7f380c33f56986aefb7c83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([2994, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2994, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2994, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2994, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4fa2002e6d7f380c33f56986aefb7c83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([2994, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2994, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2994, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2994, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d5c47e276a7632be0d355052650efdf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1a56bdf6beaf3fd1db44864a0b840a4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([3854, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3854, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3854, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3854, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1a56bdf6beaf3fd1db44864a0b840a4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([3854, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3854, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3854, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3854, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_72aa8d8d05456e8611725c8efe07ea3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_50b11169d450c2e8042c278fb9a30ee4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 28, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 28, 40], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4d39cebbb9ce8b420f2268eb38ce6406(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_690b381e3831f63d2c4d59a651a79570(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([100], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_7638bc76aa3f42d0111457b599af47a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 16, 16], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f6e7168907dce2d12d67d4fbce67080e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 120, 200], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 120, 200], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f773928d2200a734db0124a0490e22aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1457086f8118f6ae44c94f8b23f05743
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_f3be89c0f277058286c2fbf26294dd39(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 60, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 60, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e903aa8ad1302fdfacbf8ac819f259d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3be89c0f277058286c2fbf26294dd39
    def get_inputs(self):
        return [
            paddle.uniform([22, 60, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 60, 14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4300d7297bfb51346fc8301b61699730(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60649a0a13347a8bfb315367c178d719
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 9216, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 9216, 96], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_49fb5a5c53121cbbb077d9fc04b4a1f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 8, 8], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_a083461fb796b4c517d430f84e62c368(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a725b002c841ebd7195bbdc91c89a0c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a083461fb796b4c517d430f84e62c368
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 120, 200], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 120, 200], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_a960e4682b5a2a5cd89abd5a92f7ec0c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, 152], dtype='float32'),
            paddle.static.InputSpec(shape=[100, 152], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_68a484dfb118a85792a5944760044614(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a960e4682b5a2a5cd89abd5a92f7ec0c
    def get_inputs(self):
        return [
            paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_b174ab41add4daa07f58a0cded01a098(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[50, 76], dtype='float32'),
            paddle.static.InputSpec(shape=[50, 76], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f69fe0dca67c823514b6686719aad70d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b174ab41add4daa07f58a0cded01a098
    def get_inputs(self):
        return [
            paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_2c0afdcbc127a986d0493e5e53870e69(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[25, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[25, 38], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ad4ee6922c92d223e5a85d91b92d75ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c0afdcbc127a986d0493e5e53870e69
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_55304ee151add2b9bba22e1d623bdd68(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[13, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[13, 19], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9f06df5683fd3fbc3918ebd050c18780(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55304ee151add2b9bba22e1d623bdd68
    def get_inputs(self):
        return [
            paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_117b0b37907192d9279457a2b34eae69(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[7, 10], dtype='float32'),
            paddle.static.InputSpec(shape=[7, 10], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d672b9189d715a0db51923590733cf4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_117b0b37907192d9279457a2b34eae69
    def get_inputs(self):
        return [
            paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_c5ae14ceab59c880c270d07936a63665(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[15200, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[3800, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[950, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[247, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[70, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d63d90da3d59f84d4bfcae533bc0add3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5ae14ceab59c880c270d07936a63665
    def get_inputs(self):
        return [
            paddle.uniform([15200, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([3800, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([950, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([247, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([70, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_cadbf52be64784c09df1a2d074e92f13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 20, 20], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d2d9fb0efc0941188e0fa11827d94709(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 56, 56], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_fd63e9365b1931d59162ae47dc77e0b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_eaa0b3ce2a1281c6e4185372b36da52e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 20, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 20, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_019177d611734a910ae150bc64680359(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eaa0b3ce2a1281c6e4185372b36da52e
    def get_inputs(self):
        return [
            paddle.uniform([22, 20, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 20, 28, 28], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1ff139d5e53b954fa27f5eaa97d74e14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_144ad015232a4ec9babf8748384970a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_959ccfa6b5df262a29ac321871188c90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6e49408cda69bce22d03907e6a4ad98e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_7d4b4b7a886369b5abf5c9b7038c96f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1dcedba4fb0f6e11120f19dc6dca36ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_1a77cb9f3adb8f9d9229879f5d13bfc1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 12, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 12, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e85f8c49fcfc1674f75cea147fa8b7a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a77cb9f3adb8f9d9229879f5d13bfc1
    def get_inputs(self):
        return [
            paddle.uniform([22, 12, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 12, 56, 56], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_7b863a86ec53ab56f01ee47f912f9365(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8681c1a1016d2414d6f73dfd91b7dc98
    def get_inputs(self):
        return [
            paddle.uniform([6, 2304, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 2304, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 2304, 192], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_c9b45179d377f41c8513985e207940e1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 180, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 180, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1c442672e4d473cb753011571d4b82c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9b45179d377f41c8513985e207940e1
    def get_inputs(self):
        return [
            paddle.uniform([22, 180, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 180, 14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_0e8ff1838fe6faae4e847d8c7db557b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.37080684304237366, 0.4050702452659607, 0.07015335559844971, 0.19513855874538422]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_c451a4630aa40476f8c12a2dd8b667f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_4ce3bdc5c0ba6689ceed7bffe337608f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.30841371417045593], [0.16765755414962769], [0.24464209377765656], [0.4673091173171997]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.05442650616168976], [0.35331493616104126], [0.08072906732559204], [0.19166995584964752]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.2406461387872696], [0.24191485345363617], [0.32453492283821106], [0.31000232696533203]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.48310473561286926], [0.2981303632259369], [0.07768220454454422], [0.4662059247493744]], dtype='float32').reshape([4, 1]),
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
class TestPrimitiveOp_ee4e309a1cc8d10523ac1d77a9b7f348(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.08165464550256729], [0.20990590751171112], [0.34994369745254517], [0.2936878800392151]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.20266421139240265], [0.19550111889839172], [0.2868219316005707], [0.05105381831526756]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.2287028282880783], [0.38752099871635437], [0.013475647196173668], [0.4039143919944763]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.029762178659439087], [0.26578471064567566], [0.1908850222826004], [0.07886908948421478]], dtype='float32').reshape([4, 1]),
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
class TestPrimitiveOp_68594b557b2db00ca3dd682d3e60e2e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 56, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 56, 80], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_fd63e9365b1931d59162ae47dc77e0b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_2c7087483d6a7edb614a72ef92dfc338(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6b1002cf5bd85949783ef2c25a960e4
    def get_inputs(self):
        return [
            paddle.uniform([6, 576, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 576, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 576, 384], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_49fb5a5c53121cbbb077d9fc04b4a1f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 8, 8], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1bd40d55c190d01816e2bf811d0a43f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1bd40d55c190d01816e2bf811d0a43f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_31f3771d77b3eb287bde90ee8b122515(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c1782ba106b739e0998af469f53e9819(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 30, 50], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 30, 50], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_3d4f109297c9f109d6ad3d8aeec19c0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1457086f8118f6ae44c94f8b23f05743
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_ac91c718cafc4d70d922717cfe40d71f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 196, 4, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 196, 4, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 196, 4, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0f03d9df43b037b3cb01aeda9ca8e4f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac91c718cafc4d70d922717cfe40d71f
    def get_inputs(self):
        return [
            paddle.uniform([22, 196, 4, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 196, 4, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 196, 4, 32], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_9a9a2119522180997a70051854b22786(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[72, 72], dtype='float32'),
            paddle.static.InputSpec(shape=[72, 72], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_62fccef12a052df23954020672006d85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a9a2119522180997a70051854b22786
    def get_inputs(self):
        return [
            paddle.uniform([72, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([72, 72], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_ea4e995872996f08c2e0ddf127ce8f8d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[36, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[36, 36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7a7f813f2defd56a4ec6ed0e7bd1eba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea4e995872996f08c2e0ddf127ce8f8d
    def get_inputs(self):
        return [
            paddle.uniform([36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 36], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_dcefb85597ca23d8362580812deb084a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[18, 18], dtype='float32'),
            paddle.static.InputSpec(shape=[18, 18], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9c3ed92d5802b6502c303f20d596a8a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dcefb85597ca23d8362580812deb084a
    def get_inputs(self):
        return [
            paddle.uniform([18, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([18, 18], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_805263bdb92fe1c58cb0d47d23a100c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_62fccef12a052df23954020672006d85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a9a2119522180997a70051854b22786
    def get_inputs(self):
        return [
            paddle.uniform([72, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([72, 72], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_7a7f813f2defd56a4ec6ed0e7bd1eba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea4e995872996f08c2e0ddf127ce8f8d
    def get_inputs(self):
        return [
            paddle.uniform([36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 36], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9c3ed92d5802b6502c303f20d596a8a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dcefb85597ca23d8362580812deb084a
    def get_inputs(self):
        return [
            paddle.uniform([18, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([18, 18], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_50bc6f8d04b6f3f17ca313dd335cf748(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 240, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e1a4dd60c6b56635d4c9571f0f8abad1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50bc6f8d04b6f3f17ca313dd335cf748
    def get_inputs(self):
        return [
            paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_138f0ee51b3230e45d32ed40b249ddc0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 16, 12, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 16, 12, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 16, 12, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9003d27dd0c23eb731b000709f87b3f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_138f0ee51b3230e45d32ed40b249ddc0
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 12, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 16, 12, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 16, 12, 32], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f4440d5093b18b5508df98b435bc89b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9392b30c0e30f025725ed475fb5621b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 18, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 18, 18], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_419eabf62d96aab28e5ef4fbde7c66c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f106e9dda5232c93243dff7fc2676c69
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 14, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 14, 20], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b09719fba1c8851eb89c963f8670262b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 48, 48], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d42fb3964728ec789c93a716a13d3538(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([4162, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4162, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4162, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4162, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d42fb3964728ec789c93a716a13d3538(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([4162, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4162, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4162, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4162, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_70b9d3bbbd752904d5da78c302fc4f5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_144ad015232a4ec9babf8748384970a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_959ccfa6b5df262a29ac321871188c90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6e49408cda69bce22d03907e6a4ad98e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_84e0304e79683368a747c7e7ad0dfa4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.30523911118507385, 0.28430986404418945, 0.2956557273864746, 0.14955098927021027], [0.007251692470163107, 0.2660900950431824, 0.007713637314736843, 0.4912109673023224], [0.42819029092788696, 0.22352567315101624, 0.11200693994760513, 0.27329325675964355], [0.2480764091014862, 0.19159568846225739, 0.20336610078811646, 0.28056246042251587], [0.1583784967660904, 0.2905121445655823, 0.11682926118373871, 0.2795194387435913], [0.3372688293457031, 0.11013340204954147, 0.3130094110965729, 0.12403487414121628], [0.28677043318748474, 0.2033459097146988, 0.22687122225761414, 0.37921443581581116]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_5f2a9a34d9ce095a6d373a978df49d93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([7], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_147c0b5d7e5037d328177ca3d82b7d73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.47236743569374084, 0.2363184541463852, 0.24053102731704712, 0.06922545284032822], [0.03943862393498421, 0.05250733345746994, 0.0664558932185173, 0.45546039938926697], [0.25440242886543274, 0.3270745277404785, 0.48870959877967834, 0.15874430537223816], [0.041173409670591354, 0.35864633321762085, 0.4839353859424591, 0.26651546359062195], [0.42924126982688904, 0.1876508891582489, 0.16453306376934052, 0.1503685563802719], [0.09827739000320435, 0.42054814100265503, 0.019938861951231956, 0.16301748156547546]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_3af90ab907c9af06442c56824c7738ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([6], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_b389237df845b14ae35a31353be929d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a083461fb796b4c517d430f84e62c368
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 192, 288], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_df0a61246424ee1fb2573328ee1fa968(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 49, 16, None], dtype='float32'),
            paddle.static.InputSpec(shape=[22, 49, 16, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c4d58ff896ab581cd9c89a9a50f42b0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df0a61246424ee1fb2573328ee1fa968
    def get_inputs(self):
        return [
            paddle.uniform([22, 49, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 49, 16, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_bb4ed6179b2a01843f7b0404f8c4d82d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
    def get_inputs(self):
        return [
            paddle.uniform([1, 21504, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 21504, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_88bc0c57e668f9e2df7223998fb5ca57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 36], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1dc29ff3e00161fc6a324b0d49f2136b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_cd8413f40117fe087071a90c85e9eb53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([300], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_c775f226a8b2b694c6dba8eb98444794(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 24, 36], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a81a43d3a597908b508deaae85c43d9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_4c0d35f3b5a0f30e8f44d6221d391b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([8], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_e98d67fe7d8717330a4c85cbf5658159(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.16903062164783478, 0.3801995515823364, 0.13765276968479156, 0.11802694201469421], [0.38483214378356934, 0.22154369950294495, 0.21937145292758942, 0.36948561668395996]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_97e60f0d8187517dbccc2e56bc869089(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_05b8bde67313b7d9c11fce33ea18321d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_edba7fda5fe10881888a3f3b2ba969ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 16, 16], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_144ad015232a4ec9babf8748384970a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_959ccfa6b5df262a29ac321871188c90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6e49408cda69bce22d03907e6a4ad98e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_cca586620806f9812960aa2a71b76721(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_690b381e3831f63d2c4d59a651a79570(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([100], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_3bab8179aa3694797c877d3b359028b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 112, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 112, 160], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_e659c0e79600def1c3a8d34b28716f9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 6, 6], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 6, 6], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_61f43a31ca5da36e676a6ff28ff71b21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 52, 52], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f711307187e6a459aa52e55d4d96254e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 64, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1f2eafe799c9e741dde083e76a9c66de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_240cd57f8d226b2bdfadfe5f4ae41924(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 48], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_56d69b5b0bcbfad9b50f41c85652f5e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5c828b0cc0c36d336ec79f8430d27e58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1f2eafe799c9e741dde083e76a9c66de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_240cd57f8d226b2bdfadfe5f4ae41924(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([48, 48], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_56d69b5b0bcbfad9b50f41c85652f5e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_19d110fc29f042d2c5506bd7fa43bfa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 60, 60], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4386e65e3847411e1b8a1382e02f3ddc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_082d50bed6d2d2deb2b22cb0f7e145ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 11, 11], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 11, 11], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8dd255f3b51d7694b446f247a1b75b7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.34958770871162415, 0.4914834797382355, 0.030791470780968666, 0.4862590730190277], [0.4058743715286255, 0.19737207889556885, 0.027092628180980682, 0.4146954119205475]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_97e60f0d8187517dbccc2e56bc869089(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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


class PrimitiveOp_41364762b0d90a75a6eda5186e6f0202(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2597f29bf1b5e3789e6c900f08d6c468(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41364762b0d90a75a6eda5186e6f0202
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_cf61bf4af558c4a79a30a62f8b82e16b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 30, 50], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 30, 50], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_961792c66c633a0620e72d153f6e63fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 9, 9], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 9, 9], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9c9713b123b59730ec06ab5cdb0721f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2129367263bf0a2cc9c6b900b6f92dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.20786486566066742, 0.15852090716362]], [[0.3359413743019104, 0.38385361433029175]], [[0.10500519722700119, 0.088572658598423]], [[0.05509009584784508, 0.2576507329940796]], [[0.08082795888185501, 0.03169358894228935]], [[0.14791473746299744, 0.2231413871049881]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.36782288551330566, 0.23864783346652985]], [[0.32681968808174133, 0.39968106150627136]], [[0.12844915688037872, 0.11420886218547821]], [[0.1615632325410843, 0.4442180097103119]], [[0.2168838232755661, 0.020212797448039055]], [[0.12264090776443481, 0.462643563747406]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.18216867744922638, 0.01919618621468544]], [[0.22644925117492676, 0.1858130693435669]], [[0.4714430272579193, 0.4137498438358307]], [[0.07776288688182831, 0.44226565957069397]], [[0.41090327501296997, 0.1718730330467224]], [[0.06405521184206009, 0.37997767329216003]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.48425427079200745, 0.46140798926353455]], [[0.48020419478416443, 0.3810528516769409]], [[0.0325995534658432, 0.16867685317993164]], [[0.1398477405309677, 0.08869140595197678]], [[0.20558899641036987, 0.37943241000175476]], [[0.45008981227874756, 0.07396645098924637]]]], dtype='float32').reshape([1, 6, 1, 2]),
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
class TestPrimitiveOp_63772588aea95c8fb6fb6491d67d62e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a175d0c0530a3c5162117bbcd513875
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.3828085660934448, 0.03431301191449165], [0.023540113121271133, 0.25350096821784973], [0.1122574508190155, 0.42175671458244324], [0.48164764046669006, 0.4113951027393341], [0.07737068086862564, 0.37415698170661926], [0.0703463926911354, 0.2796851694583893]]], dtype='float32').reshape([1, 6, 2]),
            paddle.to_tensor([[[0.2346169650554657, 0.24948328733444214], [0.2773289084434509, 0.49571195244789124], [0.20115098357200623, 0.48211953043937683], [0.4969460070133209, 0.4321368932723999], [0.07187119126319885, 0.46304598450660706], [0.45128872990608215, 0.42227795720100403]]], dtype='float32').reshape([1, 6, 2]),
            paddle.to_tensor([[[0.4205370247364044], [0.3870924711227417], [0.393877238035202], [0.1548210084438324], [0.3596498370170593], [0.4254036545753479]]], dtype='float32').reshape([1, 6, 1]),
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
class TestPrimitiveOp_74c6c37652caafcb9ffba3c1e0587aa4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 7, 7], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_7a32f6f08b9ebf9769240cea6e858a64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 24, 24], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ab6171e1d9ea31c6c02f10ec574dd0dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_2521abbc92df2a5034518c34fb749b60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4417053759098053, 0.45320624113082886, 0.28990837931632996, 0.4012581706047058], [0.3686635196208954, 0.4158131778240204, 0.2523808479309082, 0.48548614978790283]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_97e60f0d8187517dbccc2e56bc869089(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_43083f02af5d39f7b67d7ee0c0c0260c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a81a43d3a597908b508deaae85c43d9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_4c0d35f3b5a0f30e8f44d6221d391b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([8], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_72aa8d8d05456e8611725c8efe07ea3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_89161b457bd2a2bc6708b5c78c1eaa8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5dc53de0d2688a5b4748447c9855df4
    def get_inputs(self):
        return [
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ea47b32c79f402891afd52bdc17c1919(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([22, 40, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 40, 14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_979426c7c8aa6341f1f78e034608ae3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_12bf2ac8c534ef4d25941c2dc1369827(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 24, 24], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b608b7e54c3e9c0db3bd2ae1327a0621(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_74c6c37652caafcb9ffba3c1e0587aa4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 7, 7], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d72736ee64e7806324ddc10902ba3424(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 15, 15], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_58ff7aa29e7c894b62adecece32e7017(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_45753d63be77e5a306508a8b2d5e948f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 36, 36], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4386e65e3847411e1b8a1382e02f3ddc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_51984546ca7ee8c214d275832c45dc36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([1738, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1738, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1738, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1738, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_51984546ca7ee8c214d275832c45dc36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([1738, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1738, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1738, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1738, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_05b8bde67313b7d9c11fce33ea18321d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b4ab8938db859088e381c651a072219d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 8, 8], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_70b9d3bbbd752904d5da78c302fc4f5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_632825a5440eead2cdfc0d10e827dc22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([20, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([20, 30], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_959ccfa6b5df262a29ac321871188c90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_144ad015232a4ec9babf8748384970a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f60c81c970ad1c48f24c5a7d5db3dd2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 16], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_068144b324fbc9d974dcb67738c5cad6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_959ccfa6b5df262a29ac321871188c90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_144ad015232a4ec9babf8748384970a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f60c81c970ad1c48f24c5a7d5db3dd2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 16], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_3d984df7333b2f219bb4c0544c4aeea7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.11145345121622086, 0.23862624168395996, 0.405653178691864, 0.05646820738911629], [0.4446398913860321, 0.4059602618217468, 0.3826504349708557, 0.2837061583995819], [0.06379975378513336, 0.22761933505535126, 0.051665134727954865, 0.4780203104019165], [0.1788037121295929, 0.17385177314281464, 0.37165603041648865, 0.09100950509309769], [0.21199457347393036, 0.07097514718770981, 0.23687568306922913, 0.35021036863327026], [0.029423732310533524, 0.006402377039194107, 0.20529043674468994, 0.2994297742843628], [0.13569502532482147, 0.2893596589565277, 0.11089305579662323, 0.09220270812511444]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_5f2a9a34d9ce095a6d373a978df49d93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([7], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_11c9f70cbe998840308baa5ba8c1dbd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 48, 48], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b38f7c0d5748f90ac42d1cfceb596f6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 48, 72], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_05b8bde67313b7d9c11fce33ea18321d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_dda2a47081d8eb070b1881dc1fd6993a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34b5bfe351392a61a4919e8b0fb9b096
    def get_inputs(self):
        return [
            paddle.uniform([22, 28, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 28, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 28, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 28, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 28, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 28, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 28, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 28, 28, 28], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_46504505bc8b51184e0493f57585a41d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3228645920753479], [0.3960472047328949], [0.3066532611846924], [0.021218301728367805], [0.2319512814283371], [0.3718463182449341], [0.08288423717021942], [0.33881354331970215], [0.38703471422195435]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.22029320895671844], [0.11446359008550644], [0.09660761803388596], [0.22756057977676392], [0.20094726979732513], [0.00021650124108418822], [0.06975062936544418], [0.1015438437461853], [0.14690779149532318]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.3899255692958832], [0.01601838320493698], [0.2937104105949402], [0.18330015242099762], [0.02360283024609089], [0.38878753781318665], [0.2861289083957672], [0.20783261954784393], [0.16109074652194977]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.0589759387075901], [0.4124438464641571], [0.06439685821533203], [0.264473021030426], [0.11398261785507202], [0.15532927215099335], [0.11311130225658417], [0.4885890781879425], [0.38761046528816223]], dtype='float32').reshape([9, 1]),
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
class TestPrimitiveOp_3417fe1fd707f662f34efad197e2446d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4389498233795166], [0.15551380813121796], [0.15075920522212982], [0.32667702436447144], [0.1646711528301239], [0.46801039576530457], [0.43087238073349], [0.4495995342731476], [0.2844972312450409]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.01866067200899124], [0.43595606088638306], [0.27865275740623474], [0.24296759068965912], [0.07518929243087769], [0.087107814848423], [0.4223915636539459], [0.44737523794174194], [0.3034294545650482]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.018780270591378212], [0.3621695339679718], [0.48838600516319275], [0.15385977923870087], [0.4510036110877991], [0.35555529594421387], [0.32748493552207947], [0.21274198591709137], [0.13382747769355774]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.19809138774871826], [0.1797289103269577], [0.031369950622320175], [0.1931517869234085], [0.29963502287864685], [0.2445525825023651], [0.03370177373290062], [0.06182239204645157], [0.3453497588634491]], dtype='float32').reshape([9, 1]),
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
class TestPrimitiveOp_cbc4d7cd667c46d47f21de001c4e895c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 60, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 60, 100], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6933fc52eaa223c14c2741bab6c14bb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a584fcd652d89ea075bfce9f2062e1b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f395f03fafee203c555a94643e375026(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f6d6a9ddb8f95fbf71f1c24be8d10219(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 16, 32], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_7778eed994216c0e5523a822369834e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 4, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_05259161f044cc3fee930f1312826aab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([5553, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5553, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5553, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5553, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_05259161f044cc3fee930f1312826aab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([5553, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5553, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5553, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5553, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b5f7f362f5ad5b569ecb7d82e6494e97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ae021050230c6fdaa1d738576612010d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_7f3b97fa22d7ae363c56a7824a8e8868(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.024022752419114113, 0.38020074367523193, 0.34132951498031616, 0.44830092787742615], [0.23893548548221588, 0.15022563934326172, 0.3468237817287445, 0.3618110716342926], [0.41037508845329285, 0.2736871540546417, 0.4043949544429779, 0.17460180819034576], [0.2557991147041321, 0.2483685314655304, 0.19695279002189636, 0.13544489443302155], [0.08377285301685333, 0.4270392656326294, 0.4616359770298004, 0.27932026982307434], [0.4875655174255371, 0.26046931743621826, 0.03341309726238251, 0.3536261022090912]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_3af90ab907c9af06442c56824c7738ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([6], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_cbc4d7cd667c46d47f21de001c4e895c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 60, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 60, 100], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_0257890e925a139b16372a5d1da55ec8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 28, 28], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6dec380f21e65c9fcb0a4d619a1cc373(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 8, 112, 112], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_90c72ac2fc530513aef7cd3da0785a30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b568976cdadaa0516048d1c89b88cea8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2717084288597107, 0.45454221963882446, 0.006341543979942799, 0.13451623916625977], [0.47601041197776794, 0.38216322660446167, 0.012220052070915699, 0.04670698568224907], [0.3812931478023529, 0.37026315927505493, 0.3200012743473053, 0.1258433759212494]], dtype='float32').reshape([3, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_ddc5f29dbd04067f08de4222dfec09e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([3], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_4a1124fc591ac9867819bf737c738110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 34, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 34, 34], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_17043484afab04c36688a2001e6f7b9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 7, 10], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_7ed4049586f80e8798ec3607daf8ed92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([22, 120, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 120, 28, 28], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_dea896ef235528715b6a2535f02076e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 44, 44], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 44, 44], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f0740cf76dfcc5c2cbf7abd4fdc48a1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34b5bfe351392a61a4919e8b0fb9b096
    def get_inputs(self):
        return [
            paddle.uniform([22, 28, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 28, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 28, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 28, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 28, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 28, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 28, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 28, 56, 56], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d4bb432e730ad8ef66231d92c5203859(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 18, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 18, 18], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_499fbde81d3808b6ea0a46a48f40c6f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 17, 17], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 17, 17], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f607bff1041daf3617005641d1e80184(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([22, 120, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 120, 14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9e1538baf0702a5a965023a2fbe99923(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34b5bfe351392a61a4919e8b0fb9b096
    def get_inputs(self):
        return [
            paddle.uniform([10, 28, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 28, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 28, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 28, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 28, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 28, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 28, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 28, 56, 56], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_21353360348922b803268b71dc2366a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 22, 22], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 22, 22], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_14792fffd2c65a7dd2ffa3464aa72e3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5dc53de0d2688a5b4748447c9855df4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.48949873447418213, 0.2125146985054016, 0.3943987786769867, 0.052135169506073, 0.3060843348503113, 0.14483611285686493], dtype='float32').reshape([6]),
            paddle.to_tensor([0.1455189287662506, 0.18382209539413452, 0.022990306839346886, 0.4498473107814789, 0.47428104281425476, 0.17863892018795013], dtype='float32').reshape([6]),
            paddle.to_tensor([0.033982645720243454, 0.30670925974845886, 0.2704288363456726, 0.40632981061935425, 0.399911105632782, 0.4827656149864197], dtype='float32').reshape([6]),
            paddle.to_tensor([0.46267789602279663, 0.06502489745616913, 0.23793815076351166, 0.4408537447452545, 0.28138476610183716, 0.4201972782611847], dtype='float32').reshape([6]),
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
class TestPrimitiveOp_3d7a6da84e68db53fff9a848dc937ca6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5dc53de0d2688a5b4748447c9855df4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3529529273509979, 0.1577829271554947, 0.3188677430152893, 0.33222490549087524, 0.2642192542552948, 0.17672815918922424], dtype='float32').reshape([6]),
            paddle.to_tensor([0.2904326915740967, 0.35068219900131226, 0.13571391999721527, 0.24345217645168304, 0.447722464799881, 0.3805902302265167], dtype='float32').reshape([6]),
            paddle.to_tensor([0.4372062087059021, 0.4139530062675476, 0.03780849650502205, 0.035732731223106384, 0.03976374492049217, 0.21069513261318207], dtype='float32').reshape([6]),
            paddle.to_tensor([0.28935226798057556, 0.455740362405777, 0.22648458182811737, 0.2802315354347229, 0.46797147393226624, 0.46781590580940247], dtype='float32').reshape([6]),
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
class TestPrimitiveOp_c1775970acdae735c157661e8a52eacf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([1733, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1733, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1733, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1733, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c1775970acdae735c157661e8a52eacf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([1733, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1733, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1733, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1733, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_05b8bde67313b7d9c11fce33ea18321d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_fa0722a8bef29126a25f5a0594a17d14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 5, 5], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 5, 5], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_aae150eb0fc942f51c08117b5ecb1d71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a175d0c0530a3c5162117bbcd513875
    def get_inputs(self):
        return [
            paddle.uniform([4, 144, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 144, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 144, 768], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4386e65e3847411e1b8a1382e02f3ddc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6933fc52eaa223c14c2741bab6c14bb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a584fcd652d89ea075bfce9f2062e1b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f395f03fafee203c555a94643e375026(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_43083f02af5d39f7b67d7ee0c0c0260c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6f89770c08b8df6bca6debfbfafb8e4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.08432912081480026, 0.4075654149055481, 0.42063939571380615, 0.2695857286453247], [0.06464184820652008, 0.2741512060165405, 0.11042358726263046, 0.23393476009368896]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_97e60f0d8187517dbccc2e56bc869089(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([2], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_31f3771d77b3eb287bde90ee8b122515(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_776d703246949de25cd2e9dd927c7b3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.30835163593292236, 0.07124757021665573, 0.18481184542179108, 0.10005176812410355]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_c451a4630aa40476f8c12a2dd8b667f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_a00e76e3ff79a2bd5e025d26b6a16aa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 17, 128, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_30cc89538f842dbc1ec95e3f6bef6063(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b9dea30d0cbeb10588a5bd833f33b1e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([80, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 80], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_373feead1a9fbbed37952fe0804f5e40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([40, 40], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_bcbe28e378696ed8518cfa7a8a7b580f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([20, 20], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_70b9d3bbbd752904d5da78c302fc4f5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b9dea30d0cbeb10588a5bd833f33b1e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([80, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 80], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_373feead1a9fbbed37952fe0804f5e40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([40, 40], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_bcbe28e378696ed8518cfa7a8a7b580f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([20, 20], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_47ff3884c46ce473bca1559ef26f47ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41364762b0d90a75a6eda5186e6f0202
    def get_inputs(self):
        return [
            paddle.uniform([22, 49, 8, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 49, 8, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 49, 8, 32], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d6ceac466e65ed955a6a19d648edaff9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 13, 13], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 13, 13], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ee48c8e4f066a56ab10cb0112564321f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41364762b0d90a75a6eda5186e6f0202
    def get_inputs(self):
        return [
            paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4c7e8a203eec421bfdb121d3019855c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 36, 36], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a51d1c57ef055c27da9ed5f2f70ba499(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a88937a4ba4b9baec3388c0967dd94bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34b5bfe351392a61a4919e8b0fb9b096
    def get_inputs(self):
        return [
            paddle.uniform([22, 112, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 112, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 112, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 112, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 112, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 112, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 112, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 112, 14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_e547a28ba2aa7721d42891e822f7d708(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_11c9f70cbe998840308baa5ba8c1dbd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 48, 48], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_144ad015232a4ec9babf8748384970a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_959ccfa6b5df262a29ac321871188c90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6e49408cda69bce22d03907e6a4ad98e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d5c47e276a7632be0d355052650efdf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6797e81d7bfcd1d1fc47009bab3eca6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([1466, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1466, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1466, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1466, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6797e81d7bfcd1d1fc47009bab3eca6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([1466, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1466, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1466, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1466, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4233c14a8059d68bd80d489a3d3ae3a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8c5f3c7843830a001e0e98f85b5f618f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 30, 30], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_77bc556f36c2162ce7909f25b18d7ccb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34b5bfe351392a61a4919e8b0fb9b096
    def get_inputs(self):
        return [
            paddle.uniform([22, 56, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 56, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 56, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 56, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 56, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 56, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 56, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 56, 28, 28], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4f74f401cc2fe46e6eb5c360035e293b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.21787340939044952, 0.3058196008205414, 0.1441837102174759, 0.4488373398780823], [0.41595470905303955, 0.12608909606933594, 0.16094008088111877, 0.10529551655054092], [0.17940425872802734, 0.05010313168168068, 0.2323574274778366, 0.29729440808296204], [0.2608910799026489, 0.2676137089729309, 0.11929520219564438, 0.0292961522936821], [0.17230208218097687, 0.19723495841026306, 0.4367802143096924, 0.26049256324768066], [0.2738981544971466, 0.11974108219146729, 0.27832460403442383, 0.24150776863098145], [0.21795953810214996, 0.30982330441474915, 0.22795483469963074, 0.3849623203277588]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_5f2a9a34d9ce095a6d373a978df49d93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([7], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_929b941f589f3c0444f10fecdc1374c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.14669905602931976, 0.11785608530044556, 0.31231316924095154, 0.007499780505895615]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_c451a4630aa40476f8c12a2dd8b667f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_5047874ce14665dd3653812dce54ca6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34b5bfe351392a61a4919e8b0fb9b096
    def get_inputs(self):
        return [
            paddle.uniform([10, 14, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 14, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 14, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 14, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 14, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 14, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 14, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([10, 14, 56, 56], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_7da23b76417507432cf3a3bcb410c56b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41364762b0d90a75a6eda5186e6f0202
    def get_inputs(self):
        return [
            paddle.uniform([22, 80, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 80, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 80, 28, 28], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_e547a28ba2aa7721d42891e822f7d708(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_57d62f97ab2dafacdcba7dcbd74f882e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.07675330340862274]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.04051472991704941]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.2655336260795593]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.06180190667510033]], dtype='float32').reshape([1, 1]),
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
class TestPrimitiveOp_5bc3f85f9c7d55ae9455c85656deef2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.399107426404953]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.11604505032300949]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.2398943156003952]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.368866503238678]], dtype='float32').reshape([1, 1]),
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
class TestPrimitiveOp_aa24ae4a2f5342072d0366048128381e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([22, 196, 8, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 196, 8, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b855b7d6743cf85f5daf27b6671d4453(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4892083704471588], [0.27091413736343384], [0.17127905786037445], [0.1577376276254654], [0.41942358016967773], [0.17888130247592926]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.21311093866825104], [0.05007264018058777], [0.38398033380508423], [0.09370165318250656], [0.32851335406303406], [0.1493993103504181]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.3479418456554413], [0.34206390380859375], [0.08347898721694946], [0.061992086470127106], [0.2510312795639038], [0.4961717128753662]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.09062915295362473], [0.18488666415214539], [0.34276267886161804], [0.2707947790622711], [0.3789082169532776], [0.28018254041671753]], dtype='float32').reshape([6, 1]),
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
class TestPrimitiveOp_646cb91487bfc6a54ae6c39fba8468f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06633955240249634], [0.18247836828231812], [0.2487204223871231], [0.44636279344558716], [0.2324966937303543], [0.3986716568470001]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.37008339166641235], [0.3361941874027252], [0.14341436326503754], [0.14076094329357147], [0.3007638156414032], [0.4232746660709381]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.49149706959724426], [0.3978572487831116], [0.2893059551715851], [0.04706785827875137], [0.09867657721042633], [0.3818233013153076]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.07251711189746857], [0.2581758201122284], [0.3393520414829254], [0.4828045070171356], [0.3792950510978699], [0.3344321846961975]], dtype='float32').reshape([6, 1]),
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
class TestPrimitiveOp_56eec34e495f7f3c2c307499df1c6cb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2129367263bf0a2cc9c6b900b6f92dc
    def get_inputs(self):
        return [
            paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_e659c0e79600def1c3a8d34b28716f9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 6, 6], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 6, 6], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d28d1ebe3696cca2ecbb0aa214b7cd23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_da2bac44525413c24a709f2eee72de2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_77b5f380ded411bbb99efe60e054cb8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_014fe8222e4aca10d895aef341bbbb40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a175d0c0530a3c5162117bbcd513875
    def get_inputs(self):
        return [
            paddle.uniform([4, 576, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 576, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 576, 384], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_fd838c0c16315be90fb5ddba34cc6997(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 32, 32], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1961ee84cfc7eb06846208534f8678b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.044476039707660675, 0.05122298002243042, 0.22665901482105255, 0.20944929122924805], [0.27527180314064026, 0.1378462016582489, 0.24943381547927856, 0.14275483787059784], [0.1291610151529312, 0.4926948547363281, 0.3651009202003479, 0.2669443190097809], [0.38224121928215027, 0.27661895751953125, 0.4803200960159302, 0.21076464653015137], [0.3033926486968994, 0.44697821140289307, 0.4121702313423157, 0.30973970890045166]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_8d33ee0f7804e19ae5fa954d05146406(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([5], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_e34674c563ce674ff0e23020ad4f81d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.153085395693779, 0.15082640945911407, 0.3437725007534027, 0.08953218162059784], [0.22880303859710693, 0.3738715946674347, 0.27926191687583923, 0.27835729718208313], [0.14348462224006653, 0.010167037136852741, 0.28419172763824463, 0.32861584424972534], [0.24178385734558105, 0.2601092755794525, 0.32565972208976746, 0.07134416699409485], [0.16305334866046906, 0.3240160644054413, 0.32681068778038025, 0.030790362507104874], [0.3719404935836792, 0.41742363572120667, 0.3284975290298462, 0.010256124660372734], [0.30644646286964417, 0.2166462540626526, 0.11804600059986115, 0.16275270283222198]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_5f2a9a34d9ce095a6d373a978df49d93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([7], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_e547a28ba2aa7721d42891e822f7d708(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4a69083ef5941f7b90a90be0c74d8eb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.17277765274047852, 0.4062096178531647, 0.13662110269069672, 0.13183529675006866], [0.1836676299571991, 0.33742737770080566, 0.0845433846116066, 0.39738979935646057], [0.18749657273292542, 0.4540046155452728, 0.3578137457370758, 0.2902390658855438], [0.29468220472335815, 0.2598723769187927, 0.4815906584262848, 0.4595033824443817], [0.3465569019317627, 0.46954914927482605, 0.09371209889650345, 0.08958199620246887], [0.17359988391399384, 0.4599875211715698, 0.41239801049232483, 0.13032981753349304], [0.3529101610183716, 0.3771044611930847, 0.2441311925649643, 0.3322638273239136]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_5f2a9a34d9ce095a6d373a978df49d93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([7], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_0e04b5c38fcf5d2371528b137bc751e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 96, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 96, 96], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1982626b5b29898e572e3cb7cce23925(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 88, 88], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 88, 88], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d411ca3c1739ecab5596fa727dd43e88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.22347691655158997, 0.21912971138954163, 0.026322772726416588, 0.28863784670829773]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_c451a4630aa40476f8c12a2dd8b667f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_1dc29ff3e00161fc6a324b0d49f2136b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_cd8413f40117fe087071a90c85e9eb53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([300], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_e36ed182d8f298a6b0709ece386d4533(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a175d0c0530a3c5162117bbcd513875
    def get_inputs(self):
        return [
            paddle.uniform([4, 2304, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 2304, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 2304, 192], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f3ca06e9d4e684c2f9d147a77d4cb5c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 80, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 80, 80], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1ea2a9aa5e97290d0c8336d72a9713cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([2052, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2052, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2052, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2052, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1ea2a9aa5e97290d0c8336d72a9713cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([2052, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2052, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2052, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2052, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_31f3771d77b3eb287bde90ee8b122515(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4db40d33bb75ea13a6fcf54ed493f21f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2129367263bf0a2cc9c6b900b6f92dc
    def get_inputs(self):
        return [
            paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_56d69b5b0bcbfad9b50f41c85652f5e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_bea112607e2311e80da5e571a9c86161(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 40, 40], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_17d05eb6382d2ea58862dbdb8e9865f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34b5bfe351392a61a4919e8b0fb9b096
    def get_inputs(self):
        return [
            paddle.uniform([22, 14, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 14, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 14, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 14, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 14, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 14, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 14, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 14, 56, 56], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_dadacece982240c8744107d28c88cd17(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_77a70d4deb03de21ad869269414b83cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dadacece982240c8744107d28c88cd17
    def get_inputs(self):
        return [
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_7a32f6f08b9ebf9769240cea6e858a64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 24, 24], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_00197b0fd9053078d950a3aa967949da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([4717, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4717, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4717, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4717, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_00197b0fd9053078d950a3aa967949da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([4717, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4717, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4717, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4717, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_78bad22fe49e21cec6a67e7126a6ddd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9694e6ff00cb6491a017396a14f18581(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 160, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 17, 160, 160], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6311ec15e87a13a412f88d561891c7c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([22, 600, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 600, 7, 7], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9108ffebd70e5cb6f478fd99b26dc80b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.019870752468705177, 0.34105750918388367, 0.08597230911254883, 0.22628436982631683], [0.06770025938749313, 0.42492666840553284, 0.3536209166049957, 0.010973446071147919], [0.47830554842948914, 0.4089871644973755, 0.4721699357032776, 0.07909690588712692], [0.12308020889759064, 0.19180642068386078, 0.40730535984039307, 0.2070724219083786], [0.33429089188575745, 0.011158177629113197, 0.08848989009857178, 0.46986207365989685]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_8d33ee0f7804e19ae5fa954d05146406(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([5], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_b38f7c0d5748f90ac42d1cfceb596f6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 48, 72], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_284bc0760f42393f5e926855b0c1a97f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([1056, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1056, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1056, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1056, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_284bc0760f42393f5e926855b0c1a97f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([1056, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1056, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1056, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1056, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c63d5f01dedb07a889a46f7cc76af9ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_43083f02af5d39f7b67d7ee0c0c0260c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f4e6ed5bdb7d782bc788cf9d057035bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 32, 32], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_871a86c50ec55a304be42675abf7d075(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 15, 25], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 15, 25], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_78bad22fe49e21cec6a67e7126a6ddd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_abe25245c08fed1e8471d926e0887bb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 96, 144], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c61df594738380170c139198793e8c14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 192, 288], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ca132ebf48f034a1a571349add59d0e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34b5bfe351392a61a4919e8b0fb9b096
    def get_inputs(self):
        return [
            paddle.uniform([22, 56, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 56, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 56, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 56, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 56, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 56, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 56, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 56, 14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_419ba98224dc93f12ae7fb3713e0baa6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a175d0c0530a3c5162117bbcd513875
    def get_inputs(self):
        return [
            paddle.uniform([6, 144, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 144, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 144, 768], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_3e5058132190a639d5d26cd388a876ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([68, 68], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 68], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_16d86ea15b393d4a480f7fa28100d85d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([34, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([34, 34], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1f945e2bca3413828b3e004f7913a906(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([17, 17], dtype='float32', min=0, max=0.5),
            paddle.uniform([17, 17], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d5c47e276a7632be0d355052650efdf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3e5058132190a639d5d26cd388a876ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([68, 68], dtype='float32', min=0, max=0.5),
            paddle.uniform([68, 68], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_16d86ea15b393d4a480f7fa28100d85d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([34, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([34, 34], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1f945e2bca3413828b3e004f7913a906(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([17, 17], dtype='float32', min=0, max=0.5),
            paddle.uniform([17, 17], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_fa33e9edb6d923d6a9aafbf9c05001e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.121558278799057], [0.026957757771015167], [0.3154665231704712], [0.13702236115932465], [0.13035985827445984]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.06055105850100517], [0.3023223578929901], [0.29442358016967773], [0.12275972962379456], [0.3305083215236664]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.23420564830303192], [0.1628895252943039], [0.17040301859378815], [0.04161243140697479], [0.31444212794303894]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.12737542390823364], [0.2753457725048065], [0.42593345046043396], [0.11517052352428436], [0.06334847956895828]], dtype='float32').reshape([5, 1]),
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
class TestPrimitiveOp_5002644b2a522fec502c5c3840d4ddd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.29517462849617004], [0.09762095659971237], [0.39388298988342285], [0.45222628116607666], [0.37068048119544983]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.015375080518424511], [0.07627338171005249], [0.2160526067018509], [0.0893891453742981], [0.40267425775527954]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.20128846168518066], [0.1977842003107071], [0.1439625471830368], [0.07181569188833237], [0.2501826286315918]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.2253778576850891], [0.03415966406464577], [0.43615180253982544], [0.47151103615760803], [0.14356163144111633]], dtype='float32').reshape([5, 1]),
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
class TestPrimitiveOp_f23e18bc2b56361f1c83e79293e8da65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.28491389751434326, 0.25443458557128906, 0.024227358400821686, 0.31417572498321533]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_c451a4630aa40476f8c12a2dd8b667f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_8ee462787e16133403b927ba886fa509(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 68, 68], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 68, 68], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_abe25245c08fed1e8471d926e0887bb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 96, 144], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6e49408cda69bce22d03907e6a4ad98e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_959ccfa6b5df262a29ac321871188c90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_144ad015232a4ec9babf8748384970a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f60c81c970ad1c48f24c5a7d5db3dd2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 16], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_379b64667ba43ad8a6303d8f433aa279(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([8, 8], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_63cb58fa23a3f6d184ec8519e1128f95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 72, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 72, 72], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_3a46aebaabcd91c85fde2923b84824f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([22, 36, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 36, 56, 56], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c63d5f01dedb07a889a46f7cc76af9ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_856ee7559a12dee65d21e7b25fc57236(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a175d0c0530a3c5162117bbcd513875
    def get_inputs(self):
        return [
            paddle.uniform([4, 9216, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 9216, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([4, 9216, 96], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_715d31ab33200caceb33d78602eb679e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_0c70a7308ee0364432ec65d289354604(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.10112754255533218, 0.10138582438230515, 0.4006735682487488, 0.376541405916214]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_c451a4630aa40476f8c12a2dd8b667f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_4870f1f9e7f83a33bff195cb6b8129b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([15, 25], dtype='float32', min=0, max=0.5),
            paddle.uniform([15, 25], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_14fc5f240e2cb1646d1c5230c304e9b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_34b5bfe351392a61a4919e8b0fb9b096
    def get_inputs(self):
        return [
            paddle.uniform([22, 112, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 112, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 112, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 112, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 112, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 112, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 112, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 112, 7, 7], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f3870e9d84007173fb2f3eb3dea981ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([2354, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2354, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2354, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2354, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f3870e9d84007173fb2f3eb3dea981ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([2354, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2354, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2354, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2354, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b608b7e54c3e9c0db3bd2ae1327a0621(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a51d1c57ef055c27da9ed5f2f70ba499(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a81a43d3a597908b508deaae85c43d9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_4c0d35f3b5a0f30e8f44d6221d391b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([8], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_fda1fecd5667958b82d978577448888c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([2994, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2994, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2994, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2994, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_fda1fecd5667958b82d978577448888c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([2994, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2994, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2994, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2994, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d5c47e276a7632be0d355052650efdf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_515dcc06ad48ca0f6483606ba7824e92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([3854, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3854, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3854, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3854, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_515dcc06ad48ca0f6483606ba7824e92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([3854, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3854, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3854, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3854, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_72aa8d8d05456e8611725c8efe07ea3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1e3f1ed672a768d70e7236e2eba48e55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 28, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 28, 40], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_cca586620806f9812960aa2a71b76721(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_690b381e3831f63d2c4d59a651a79570(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([100], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_4ffee4989e420692e825f235cfe14206(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 16, 16], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6043f2ed5775e2fe5e86d512dd5ec4cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 120, 200], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 120, 200], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b5f7f362f5ad5b569ecb7d82e6494e97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_56e1566c86f681cde1f2f08c29e2357f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([22, 60, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 60, 14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_fa60117ce2008ebdaf0ff637a39e04aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a175d0c0530a3c5162117bbcd513875
    def get_inputs(self):
        return [
            paddle.uniform([6, 9216, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 9216, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 9216, 96], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_55a07bccdf4f629b02a8223851564ca8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 8, 8], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1d28e17ba3d110c98ad3d3fcbc282d5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 120, 200], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 120, 200], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ee1367c7bf949c3dc1f32605fa8f568c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
            paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4d24e06693772dd9aa359672efd4e391(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
            paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_979426c7c8aa6341f1f78e034608ae3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_e4a9bfed4497c1a883828138252bdc96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
            paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d7fb66e666b52b25a9cf3332d74459d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_1d4039695f3562fdf95293b9ad4f83a7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4]
        return input_0

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_aeebba240e2521aa78ec7a295229e9e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d4039695f3562fdf95293b9ad4f83a7
    def get_inputs(self):
        return [
            paddle.uniform([15200, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([3800, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([950, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([247, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([70, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_cadbf52be64784c09df1a2d074e92f13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 20, 20], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4a3d21c9c23d377f912866c3f5c0347a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 56, 56], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ae021050230c6fdaa1d738576612010d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_129481c0f4a5ab099c4bd60676ef6c7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([22, 20, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 20, 28, 28], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b47fc31c8da2676010f14764861aa462(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_144ad015232a4ec9babf8748384970a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_959ccfa6b5df262a29ac321871188c90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6e49408cda69bce22d03907e6a4ad98e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ab6171e1d9ea31c6c02f10ec574dd0dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a51d1c57ef055c27da9ed5f2f70ba499(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_7afb92e405eb5eb06ec82660d818de6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([22, 12, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 12, 56, 56], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5ef31bd6ee7d4c1ba6cabdcde218f435(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a175d0c0530a3c5162117bbcd513875
    def get_inputs(self):
        return [
            paddle.uniform([6, 2304, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 2304, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 2304, 192], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9028ffbe315a6dbd44c2306d17844906(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([22, 180, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 180, 14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9cb1b945d6c1dec2bd019bda2d8b48a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.37080684304237366, 0.4050702452659607, 0.07015335559844971, 0.19513855874538422]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_c451a4630aa40476f8c12a2dd8b667f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_4ce3bdc5c0ba6689ceed7bffe337608f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.30841371417045593], [0.16765755414962769], [0.24464209377765656], [0.4673091173171997]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.05442650616168976], [0.35331493616104126], [0.08072906732559204], [0.19166995584964752]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.2406461387872696], [0.24191485345363617], [0.32453492283821106], [0.31000232696533203]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.48310473561286926], [0.2981303632259369], [0.07768220454454422], [0.4662059247493744]], dtype='float32').reshape([4, 1]),
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
class TestPrimitiveOp_ee4e309a1cc8d10523ac1d77a9b7f348(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.08165464550256729], [0.20990590751171112], [0.34994369745254517], [0.2936878800392151]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.20266421139240265], [0.19550111889839172], [0.2868219316005707], [0.05105381831526756]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.2287028282880783], [0.38752099871635437], [0.013475647196173668], [0.4039143919944763]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.029762178659439087], [0.26578471064567566], [0.1908850222826004], [0.07886908948421478]], dtype='float32').reshape([4, 1]),
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
class TestPrimitiveOp_514e19c9737cb70487638acb2aece79b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 56, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 56, 80], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ae021050230c6fdaa1d738576612010d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f3ac1a218a0edb0ea3abdb0a18ad5c3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a175d0c0530a3c5162117bbcd513875
    def get_inputs(self):
        return [
            paddle.uniform([6, 576, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 576, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([6, 576, 384], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_55a07bccdf4f629b02a8223851564ca8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 8, 8], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_dcca0328200e59c0352e18d7a3cf5721(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_dcca0328200e59c0352e18d7a3cf5721(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_31f3771d77b3eb287bde90ee8b122515(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_cf61bf4af558c4a79a30a62f8b82e16b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 30, 50], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 30, 50], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4233c14a8059d68bd80d489a3d3ae3a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b94255054c2a003ede54f61a4dce5267(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41364762b0d90a75a6eda5186e6f0202
    def get_inputs(self):
        return [
            paddle.uniform([22, 196, 4, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 196, 4, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 196, 4, 32], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f4d7fcc96d9d3a873694636b98456990(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([72, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([72, 72], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_caaecd4c0edc0e3690ae17e823aa4c76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 36], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_de783c1c948fa237947766dc25da6dce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([18, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([18, 18], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_805263bdb92fe1c58cb0d47d23a100c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f4d7fcc96d9d3a873694636b98456990(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([72, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([72, 72], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_caaecd4c0edc0e3690ae17e823aa4c76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36, 36], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_de783c1c948fa237947766dc25da6dce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([18, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([18, 18], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_0e1194733ca52015955f5b7f5a03be9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_96ca494ba9ac9edbada9de7554cd80d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41364762b0d90a75a6eda5186e6f0202
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 12, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 16, 12, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 16, 12, 32], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_3fcd0b0eaa8cefc8603c73bd42fe3dfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 18, 18], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 18, 18], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4b98b19fca2e17973d4f230a12ba61dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 14, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 14, 20], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b09719fba1c8851eb89c963f8670262b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 48, 48], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 48, 48], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_bdc7e6391f7e331ba3deca7f633616c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([4162, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4162, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4162, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4162, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_bdc7e6391f7e331ba3deca7f633616c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([4162, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4162, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4162, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4162, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_70b9d3bbbd752904d5da78c302fc4f5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1848e60d49960e74b723e5f73421c99d
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_144ad015232a4ec9babf8748384970a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_959ccfa6b5df262a29ac321871188c90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6e49408cda69bce22d03907e6a4ad98e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9928c1d7db2e4cadcefca5f88e7e5150
    def get_inputs(self):
        return [
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_56e336e67a1a60a31174ed7de84a1c09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.30523911118507385, 0.28430986404418945, 0.2956557273864746, 0.14955098927021027], [0.007251692470163107, 0.2660900950431824, 0.007713637314736843, 0.4912109673023224], [0.42819029092788696, 0.22352567315101624, 0.11200693994760513, 0.27329325675964355], [0.2480764091014862, 0.19159568846225739, 0.20336610078811646, 0.28056246042251587], [0.1583784967660904, 0.2905121445655823, 0.11682926118373871, 0.2795194387435913], [0.3372688293457031, 0.11013340204954147, 0.3130094110965729, 0.12403487414121628], [0.28677043318748474, 0.2033459097146988, 0.22687122225761414, 0.37921443581581116]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_5f2a9a34d9ce095a6d373a978df49d93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([7], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_3c7fc02cc2cda1810c46d751cd541099(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.47236743569374084, 0.2363184541463852, 0.24053102731704712, 0.06922545284032822], [0.03943862393498421, 0.05250733345746994, 0.0664558932185173, 0.45546039938926697], [0.25440242886543274, 0.3270745277404785, 0.48870959877967834, 0.15874430537223816], [0.041173409670591354, 0.35864633321762085, 0.4839353859424591, 0.26651546359062195], [0.42924126982688904, 0.1876508891582489, 0.16453306376934052, 0.1503685563802719], [0.09827739000320435, 0.42054814100265503, 0.019938861951231956, 0.16301748156547546]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
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
class TestPrimitiveOp_3af90ab907c9af06442c56824c7738ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c94fdf92a6bffb44756955b6d8df0b87
    def get_inputs(self):
        return [
            paddle.to_tensor([6], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
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
class TestPrimitiveOp_46bcbd4c4302e9485e730ae9d92c692c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 192, 288], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_529990e39fbc574e0b8f6e8e56d16b87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_092a8d62914230a463eeb9d7215936c4
    def get_inputs(self):
        return [
            paddle.uniform([22, 49, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([22, 49, 16, 64], dtype='float32', min=0, max=0.5),
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