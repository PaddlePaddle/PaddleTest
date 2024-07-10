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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c0e9cd45b5b2ca0646cac9442f305db2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2829858958721161, 0.4483707547187805, 0.4971335828304291, 0.15902025997638702], [0.18705083429813385, 0.07559647411108017, 0.44793131947517395, 0.26347318291664124]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5b579a90dc25666caacf5c68eb623d00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2101910561323166, 0.31460368633270264, 0.09340055286884308, 0.4286078214645386], [0.12155845016241074, 0.3548726439476013, 0.3339637219905853, 0.0438280887901783]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
class TestPrimitiveOp_b088b551af67e363d473c1f13305ab4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2129367263bf0a2cc9c6b900b6f92dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.0826803669333458, 0.21053451299667358]], [[0.01925661787390709, 0.21272294223308563]], [[0.20079584419727325, 0.4657025635242462]], [[0.1783876121044159, 0.43177303671836853]], [[0.4545062184333801, 0.17607779800891876]], [[0.027255026623606682, 0.22295935451984406]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.4181259572505951, 0.4688245356082916]], [[0.2953522503376007, 0.21154829859733582]], [[0.3736066520214081, 0.4246405363082886]], [[0.2786221504211426, 0.02305924892425537]], [[0.059611789882183075, 0.06799133121967316]], [[0.20997746288776398, 0.3374594748020172]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.4918944835662842, 0.1863957643508911]], [[0.34358134865760803, 0.449489951133728]], [[0.1425766795873642, 0.06280901283025742]], [[0.08684336394071579, 0.3369137942790985]], [[0.3242059051990509, 0.40161168575286865]], [[0.30807164311408997, 0.20340518653392792]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.29992616176605225, 0.2960737645626068]], [[0.1871715933084488, 0.17506901919841766]], [[0.2953713834285736, 0.05570122227072716]], [[0.0410512313246727, 0.3668789565563202]], [[0.1386876106262207, 0.18387745320796967]], [[0.2941557466983795, 0.010345684364438057]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
class TestPrimitiveOp_c96e284c34619b0b6f55f898c04c76ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a175d0c0530a3c5162117bbcd513875
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.39063310623168945, 0.3321778476238251], [0.12678611278533936, 0.3998766839504242], [0.45696884393692017, 0.4686964750289917], [0.19388934969902039, 0.4377021789550781], [0.3627975583076477, 0.09379509091377258], [0.3976432979106903, 0.031102994456887245]]], dtype='float32').reshape([1, 6, 2]),
            paddle.to_tensor([[[0.4071896970272064, 0.13613902032375336], [0.06217319518327713, 0.3260059058666229], [0.290440171957016, 0.19214685261249542], [0.2717708647251129, 0.45718052983283997], [0.47337135672569275, 0.2392512559890747], [0.26854974031448364, 0.45605215430259705]]], dtype='float32').reshape([1, 6, 2]),
            paddle.to_tensor([[[0.1989871859550476], [0.4671623706817627], [0.08524427562952042], [0.41312164068222046], [0.46589550375938416], [0.3238028585910797]]], dtype='float32').reshape([1, 6, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_785c1a0a9e011d370de9c652dadeb439(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.36205634474754333, 0.24116620421409607, 0.015938179567456245, 0.359531044960022], [0.022833826020359993, 0.1722264289855957, 0.4449290931224823, 0.04115581884980202]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
class TestPrimitiveOp_4f3ee59c61e2c4f17536aa822c73f094(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([1755, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1755, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1755, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1755, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4f3ee59c61e2c4f17536aa822c73f094(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([1755, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1755, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1755, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1755, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_53d34d11422698348eb182d4873c385f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2960699796676636, 0.47269707918167114, 0.12119115889072418, 0.43879640102386475], [0.4274027645587921, 0.05073362961411476, 0.2670670449733734, 0.034451622515916824], [0.07949822396039963, 0.483990341424942, 0.023792315274477005, 0.35055384039878845], [0.2663685381412506, 0.4108807444572449, 0.3508042097091675, 0.31504857540130615], [0.16125237941741943, 0.21028673648834229, 0.4028345048427582, 0.3224439024925232], [0.2329816222190857, 0.38855111598968506, 0.4902689456939697, 0.017628690227866173], [0.28133541345596313, 0.292389452457428, 0.3052775263786316, 0.45975494384765625]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
class TestPrimitiveOp_916c3eac8830d2601aa865a08e76dbef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3411575257778168], [0.487751841545105], [0.0991593450307846], [0.005567505490034819], [0.4831579327583313], [0.3312598466873169], [0.3520636558532715], [0.29121044278144836], [0.12009581178426743]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.07136072963476181], [0.18888725340366364], [0.21746514737606049], [0.009641475975513458], [0.34512582421302795], [0.023270219564437866], [0.2941991984844208], [0.3037635385990143], [0.4153595566749573]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.3918236494064331], [0.17079667747020721], [0.2618381679058075], [0.10360092669725418], [0.2927932143211365], [0.150027334690094], [0.08460388332605362], [0.4861544370651245], [0.19172121584415436]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.18829287588596344], [0.34305354952812195], [0.08177930116653442], [0.3895333707332611], [0.46782585978507996], [0.33204638957977295], [0.23466411232948303], [0.13807879388332367], [0.03301508352160454]], dtype='float32').reshape([9, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_80df38937648711249bad4e8e17d1bdf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.35281503200531006], [0.18959030508995056], [0.45423388481140137], [0.15740129351615906], [0.15308204293251038], [0.25702497363090515], [0.2365979701280594], [0.4242403507232666], [0.4313344359397888]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.25974467396736145], [0.4788047671318054], [0.053342241793870926], [0.0840245708823204], [0.10440663248300552], [0.3625282645225525], [0.09382373839616776], [0.4737075865268707], [0.38082581758499146]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.13827703893184662], [0.1286333203315735], [0.22485984861850739], [0.15737760066986084], [0.2778044044971466], [0.15569062530994415], [0.15203706920146942], [0.4001633822917938], [0.4737466871738434]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.068621426820755], [0.3688132166862488], [0.14095377922058105], [0.1812393218278885], [0.4349217712879181], [0.4618573486804962], [0.2685330808162689], [0.38841864466667175], [0.36464715003967285]], dtype='float32').reshape([9, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_57adf4fdb305c9bd659391be17b91ad7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([5589, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5589, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5589, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5589, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_57adf4fdb305c9bd659391be17b91ad7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([5589, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5589, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5589, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5589, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bac21a6486d3d23db2d5c13478dc9adf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.48915353417396545, 0.0599832721054554, 0.31578874588012695, 0.32014355063438416], [0.1614276021718979, 0.09353192150592804, 0.2042657434940338, 0.286808580160141], [0.4284842610359192, 0.20335653424263, 0.32612958550453186, 0.062386274337768555], [0.3643060624599457, 0.32442256808280945, 0.11638986319303513, 0.4475005269050598], [0.12722837924957275, 0.009367053396999836, 0.056048035621643066, 0.46019765734672546], [0.0880192220211029, 0.1442183554172516, 0.2839473783969879, 0.3002561628818512]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_04b809d141cd19242ae8b99c82719549(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.11336662620306015, 0.3186884820461273, 0.004312401637434959, 0.19526925683021545], [0.34463292360305786, 0.10320387780666351, 0.26624009013175964, 0.32023903727531433], [0.4841420650482178, 0.38910916447639465, 0.02423565462231636, 0.022353937849402428]], dtype='float32').reshape([3, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
class TestPrimitiveOp_3aeb9f7ca1efbf7a4deb5fd98c19a015(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5dc53de0d2688a5b4748447c9855df4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.41314512491226196, 0.10803968459367752, 0.3724486231803894, 0.36033663153648376, 0.18769671022891998, 0.29229438304901123], dtype='float32').reshape([6]),
            paddle.to_tensor([0.10805502533912659, 0.01880381442606449, 0.22303974628448486, 0.19843772053718567, 0.027361908927559853, 0.16955791413784027], dtype='float32').reshape([6]),
            paddle.to_tensor([0.25822827219963074, 0.04662436991930008, 0.12753066420555115, 0.47527799010276794, 0.1672964096069336, 0.08656934648752213], dtype='float32').reshape([6]),
            paddle.to_tensor([0.08468656241893768, 0.306545615196228, 0.11176025122404099, 0.0717325508594513, 0.04547895863652229, 0.05067944526672363], dtype='float32').reshape([6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b141f1f0e296cf19c8c37efe8084fe4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5dc53de0d2688a5b4748447c9855df4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.029645506292581558, 0.004914592020213604, 0.09595445543527603, 0.39793166518211365, 0.4357508420944214, 0.4137418866157532], dtype='float32').reshape([6]),
            paddle.to_tensor([0.3976886570453644, 0.18534445762634277, 0.01889265514910221, 0.45858362317085266, 0.20388272404670715, 0.15198396146297455], dtype='float32').reshape([6]),
            paddle.to_tensor([0.23812781274318695, 0.19943611323833466, 0.06108584254980087, 0.09973878413438797, 0.3285718858242035, 0.48215657472610474], dtype='float32').reshape([6]),
            paddle.to_tensor([0.10711614042520523, 0.4060676693916321, 0.3443622589111328, 0.2741973400115967, 0.03332651033997536, 0.2714977264404297], dtype='float32').reshape([6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4522fdbb64ffbb90b93b4e8bf2158cd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([1790, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1790, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1790, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1790, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4522fdbb64ffbb90b93b4e8bf2158cd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([1790, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1790, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1790, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1790, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_48209c996f11ffc359cc207b7859ac60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.053108278661966324, 0.2642962634563446, 0.12306977063417435, 0.15607690811157227], [0.0833698958158493, 0.16468219459056854, 0.2770980894565582, 0.10666385293006897]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b61d44130a2b76f9f855b14c31c225ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06575730443000793, 0.3563782870769501, 0.06684133410453796, 0.21661213040351868]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3dccc10c0a727fd7ede90e2d4929398c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([1512, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1512, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1512, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1512, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3dccc10c0a727fd7ede90e2d4929398c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([1512, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1512, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1512, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1512, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e7f557d1ef7598bd91684292e2c79fab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.009746660478413105, 0.017361877486109734, 0.4770163595676422, 0.4977247714996338], [0.32846638560295105, 0.3449159860610962, 0.01744374819099903, 0.15449169278144836], [0.4994213879108429, 0.13463035225868225, 0.3727863132953644, 0.2907992899417877], [0.1041017398238182, 0.3148060441017151, 0.3109146058559418, 0.41507911682128906], [0.28564050793647766, 0.13475272059440613, 0.14990156888961792, 0.2785126864910126], [0.022030318155884743, 0.2485632300376892, 0.45357468724250793, 0.15473639965057373], [0.44435209035873413, 0.1585448682308197, 0.25553658604621887, 0.49046581983566284]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fb86c50bde55563d62b91eb907478fa1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.27415943145751953, 0.3107362985610962, 0.3970583975315094, 0.058175958693027496]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a721fb4675d5a97272d47f829b3988fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.30021312832832336]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.29566285014152527]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.4233570992946625]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.07360603660345078]], dtype='float32').reshape([1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_94bafbec21d4833dc21c3c503d21e745(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.17907217144966125]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.4305429756641388]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.11320116370916367]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.11482519656419754]], dtype='float32').reshape([1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b9a51fe9518fbe49d8b93dcbf8a96722(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.129848912358284], [0.02661607600748539], [0.27324140071868896], [0.39769837260246277], [0.01840292103588581], [0.15425024926662445]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.11104662716388702], [0.015465895645320415], [0.3915401101112366], [0.23462030291557312], [0.08867214620113373], [0.3940161466598511]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.03926978260278702], [0.08563093096017838], [0.3721535801887512], [0.18568728864192963], [0.22553320229053497], [0.4946292042732239]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.1343114972114563], [0.39124366641044617], [0.26737356185913086], [0.42793944478034973], [0.0809512734413147], [0.09805627912282944]], dtype='float32').reshape([6, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_98d1299bf6d9b436eb683671e36c106a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.36462271213531494], [0.15699154138565063], [0.009385272860527039], [0.49828392267227173], [0.1709592342376709], [0.18620306253433228]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.07934290170669556], [0.13667255640029907], [0.20768995583057404], [0.40495753288269043], [0.464950293302536], [0.4474965035915375]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.12608090043067932], [0.20497149229049683], [0.36897599697113037], [0.33328765630722046], [0.14978736639022827], [0.34729060530662537]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.07382135838270187], [0.4149003326892853], [0.2742626368999481], [0.411622017621994], [0.4005337357521057], [0.1369883418083191]], dtype='float32').reshape([6, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_31d73ec0ddd5c9e06babf5fd7998270b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.12236736714839935, 0.023757170885801315, 0.2753857970237732, 0.0634797066450119], [0.3817211091518402, 0.3683093190193176, 0.09295232594013214, 0.3337242603302002], [0.1296713650226593, 0.17170575261116028, 0.23934559524059296, 0.2605445683002472], [0.33447957038879395, 0.19933679699897766, 0.28776758909225464, 0.13815784454345703], [0.4471212327480316, 0.07884528487920761, 0.03610456734895706, 0.03677848353981972]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4152d064a97f5fc197bfc00940806ce2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3657360374927521, 0.4700203537940979, 0.4082726836204529, 0.07482847571372986], [0.3315252363681793, 0.054085876792669296, 0.4313672184944153, 0.293525755405426], [0.27286601066589355, 0.18162350356578827, 0.05797296762466431, 0.47832754254341125], [0.07554564625024796, 0.04696071892976761, 0.2938121259212494, 0.4244130551815033], [0.2781350612640381, 0.035711824893951416, 0.16044847667217255, 0.07335866987705231], [0.126616969704628, 0.1599941849708557, 0.13273343443870544, 0.38682419061660767], [0.4670228362083435, 0.2216964066028595, 0.45244792103767395, 0.2261810600757599]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fe9392c0c1de2f27f38dcbd2d589afe8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4873305857181549, 0.3743271827697754, 0.16707535088062286, 0.4675159454345703], [0.007019971031695604, 0.4985446333885193, 0.3508499562740326, 0.226438507437706], [0.19130347669124603, 0.30717727541923523, 0.08884003758430481, 0.3845234215259552], [0.11493656039237976, 0.34949377179145813, 0.48343220353126526, 0.14999009668827057], [0.08428427577018738, 0.03870820626616478, 0.23276303708553314, 0.2538503110408783], [0.2606007158756256, 0.1668339967727661, 0.3094004988670349, 0.04491327702999115], [0.30882546305656433, 0.11893762648105621, 0.26801833510398865, 0.30436253547668457]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_816c2879d028961593166679ebed9907(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4931090474128723, 0.16415029764175415, 0.1463773250579834, 0.001705510076135397]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1d4f2ea5bd6332dd512b989309d7c2ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([2007, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2007, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2007, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2007, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1d4f2ea5bd6332dd512b989309d7c2ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([2007, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2007, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2007, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2007, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d02dcee1aa327caf058dbb1c24fafb87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([4685, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4685, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4685, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4685, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d02dcee1aa327caf058dbb1c24fafb87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([4685, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4685, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4685, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4685, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_971cc5b263fccf57f4e018729ff73cd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.08092506229877472, 0.4986841380596161, 0.3048730790615082, 0.2823871076107025], [0.4787973463535309, 0.11150021851062775, 0.2776354253292084, 0.2793109714984894], [0.2193784862756729, 0.03581467643380165, 0.47906818985939026, 0.38758349418640137], [0.2338825762271881, 0.25051364302635193, 0.43326351046562195, 0.22051788866519928], [0.1832806020975113, 0.28888657689094543, 0.3417982757091522, 0.382455974817276]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8c93084473332b85180859b4b655c366(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([1050, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1050, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1050, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1050, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8c93084473332b85180859b4b655c366(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([1050, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1050, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1050, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1050, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fa9b0901c83c9917e66ffe8b7754a21d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.323615163564682], [0.30178794264793396], [0.1656930297613144], [0.11601903289556503], [0.10725655406713486]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.2432054579257965], [0.38019073009490967], [0.11684619635343552], [0.4634474217891693], [0.16990448534488678]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.20664435625076294], [0.4538969397544861], [0.20020511746406555], [0.4132084846496582], [0.2432897984981537]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.17348651587963104], [0.32497742772102356], [0.003286341205239296], [0.16559453308582306], [0.24491815268993378]], dtype='float32').reshape([5, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1ecdb9e8b1d591a6ed2a2e0b67008768(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.13989704847335815], [0.4314858913421631], [0.14878864586353302], [0.40420448780059814], [0.3773234188556671]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.3156962990760803], [0.09945223480463028], [0.2634713351726532], [0.11271272599697113], [0.23488155007362366]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.004550704266875982], [0.4207856059074402], [0.24096810817718506], [0.004030225798487663], [0.03893037885427475]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.36014536023139954], [0.19286362826824188], [0.07153784483671188], [0.069424569606781], [0.2985104024410248]], dtype='float32').reshape([5, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4724f31d6c4e07b382da377d5a39f6bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2901870906352997, 0.0027597020380198956, 0.098127081990242, 0.32107290625572205]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_57c3b1b50186d49666a6c06a4f499c29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.15391285717487335, 0.12124611437320709, 0.10962545871734619, 0.4960550367832184]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_899cbf78b612b84f46ab5ebe97241842(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([2394, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2394, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2394, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2394, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_899cbf78b612b84f46ab5ebe97241842(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([2394, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2394, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2394, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2394, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_445b9e59a4254c8881e4437ed2f9aac1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_445b9e59a4254c8881e4437ed2f9aac1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1aa8712c6a81744438ee8becb56999d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([3758, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3758, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3758, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3758, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1aa8712c6a81744438ee8becb56999d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([3758, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3758, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3758, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3758, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dbd2994781b80aa16ef4e0cf64d8dc4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06738543510437012, 0.09491246193647385, 0.13308154046535492, 0.33248892426490784]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_947a3f1c012f98812b14ab24bde9d691(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.15036208927631378], [0.1331283152103424], [0.2937854826450348], [0.4619079828262329]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.1913178265094757], [0.294893741607666], [0.036232516169548035], [0.43063637614250183]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.06513538211584091], [0.3193783462047577], [0.09922898560762405], [0.1432037353515625]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.17245835065841675], [0.15862299501895905], [0.4525022804737091], [0.1308891922235489]], dtype='float32').reshape([4, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9fb7ed1699e3b4f5e26c5ed6884704cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.33588987588882446], [0.2792902886867523], [0.1107465997338295], [0.3460559844970703]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.2022257000207901], [0.14955280721187592], [0.286821573972702], [0.023547960445284843]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.1513662487268448], [0.28769925236701965], [0.18229663372039795], [0.12690480053424835]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.25924187898635864], [0.346165269613266], [0.1456291526556015], [0.23360179364681244]], dtype='float32').reshape([4, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7da1fd4263f92b99e76b684c586d0f28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([2008, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2008, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2008, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2008, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7da1fd4263f92b99e76b684c586d0f28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([2008, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2008, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2008, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2008, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2875dd63f259ca990d5dfce6532d19b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([4295, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4295, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4295, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4295, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2875dd63f259ca990d5dfce6532d19b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69a3489015c06588a98698e2063d474b
    def get_inputs(self):
        return [
            paddle.uniform([4295, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4295, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4295, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4295, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_877477dc3e8f6da994f519413f7ee332(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4620692729949951, 0.10149343311786652, 0.038771647959947586, 0.23243044316768646], [0.15344084799289703, 0.06268923729658127, 0.34847208857536316, 0.42999449372291565], [0.15190163254737854, 0.4104097783565521, 0.365616112947464, 0.1296401470899582], [0.27523496747016907, 0.29393917322158813, 0.0716368556022644, 0.49047335982322693], [0.012108967639505863, 0.4853935241699219, 0.2429102063179016, 0.41299933195114136], [0.030071891844272614, 0.3988513946533203, 0.48055341839790344, 0.041261013597249985], [0.1597491055727005, 0.31782782077789307, 0.05290887504816055, 0.13929510116577148]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_39b06806b6ae1112ce46ca1883e15cb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9474a7d4d5a85165199c02bcc1cf1bb7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3801068067550659, 0.3316466212272644, 0.17720377445220947, 0.15354834496974945], [0.40502336621284485, 0.17050214111804962, 0.27920740842819214, 0.3615216910839081], [0.03935656696557999, 0.3974427878856659, 0.4215141534805298, 0.31277039647102356], [0.23049986362457275, 0.42653417587280273, 0.13485263288021088, 0.48253151774406433], [0.4592631459236145, 0.49990564584732056, 0.479204922914505, 0.3003561496734619], [0.4457668662071228, 0.31206563115119934, 0.4506196677684784, 0.040063779801130295]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9c431fe29a0b1c551d9e45f351285c06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2829858958721161, 0.4483707547187805, 0.4971335828304291, 0.15902025997638702], [0.18705083429813385, 0.07559647411108017, 0.44793131947517395, 0.26347318291664124]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b50b3d2f4a0dc205ecfd832adefbd639(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2101910561323166, 0.31460368633270264, 0.09340055286884308, 0.4286078214645386], [0.12155845016241074, 0.3548726439476013, 0.3339637219905853, 0.0438280887901783]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b088b551af67e363d473c1f13305ab4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2129367263bf0a2cc9c6b900b6f92dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.0826803669333458, 0.21053451299667358]], [[0.01925661787390709, 0.21272294223308563]], [[0.20079584419727325, 0.4657025635242462]], [[0.1783876121044159, 0.43177303671836853]], [[0.4545062184333801, 0.17607779800891876]], [[0.027255026623606682, 0.22295935451984406]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.4181259572505951, 0.4688245356082916]], [[0.2953522503376007, 0.21154829859733582]], [[0.3736066520214081, 0.4246405363082886]], [[0.2786221504211426, 0.02305924892425537]], [[0.059611789882183075, 0.06799133121967316]], [[0.20997746288776398, 0.3374594748020172]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.4918944835662842, 0.1863957643508911]], [[0.34358134865760803, 0.449489951133728]], [[0.1425766795873642, 0.06280901283025742]], [[0.08684336394071579, 0.3369137942790985]], [[0.3242059051990509, 0.40161168575286865]], [[0.30807164311408997, 0.20340518653392792]]]], dtype='float32').reshape([1, 6, 1, 2]),
            paddle.to_tensor([[[[0.29992616176605225, 0.2960737645626068]], [[0.1871715933084488, 0.17506901919841766]], [[0.2953713834285736, 0.05570122227072716]], [[0.0410512313246727, 0.3668789565563202]], [[0.1386876106262207, 0.18387745320796967]], [[0.2941557466983795, 0.010345684364438057]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c96e284c34619b0b6f55f898c04c76ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a175d0c0530a3c5162117bbcd513875
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.39063310623168945, 0.3321778476238251], [0.12678611278533936, 0.3998766839504242], [0.45696884393692017, 0.4686964750289917], [0.19388934969902039, 0.4377021789550781], [0.3627975583076477, 0.09379509091377258], [0.3976432979106903, 0.031102994456887245]]], dtype='float32').reshape([1, 6, 2]),
            paddle.to_tensor([[[0.4071896970272064, 0.13613902032375336], [0.06217319518327713, 0.3260059058666229], [0.290440171957016, 0.19214685261249542], [0.2717708647251129, 0.45718052983283997], [0.47337135672569275, 0.2392512559890747], [0.26854974031448364, 0.45605215430259705]]], dtype='float32').reshape([1, 6, 2]),
            paddle.to_tensor([[[0.1989871859550476], [0.4671623706817627], [0.08524427562952042], [0.41312164068222046], [0.46589550375938416], [0.3238028585910797]]], dtype='float32').reshape([1, 6, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_10c9000bc4aad04b76a930e9cc351f69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.36205634474754333, 0.24116620421409607, 0.015938179567456245, 0.359531044960022], [0.022833826020359993, 0.1722264289855957, 0.4449290931224823, 0.04115581884980202]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a37303f775376486e864c2466d5c0b5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([1755, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1755, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1755, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1755, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a37303f775376486e864c2466d5c0b5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([1755, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1755, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1755, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1755, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_548a8b55ee7c0c1ee00289e947a0bd5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2960699796676636, 0.47269707918167114, 0.12119115889072418, 0.43879640102386475], [0.4274027645587921, 0.05073362961411476, 0.2670670449733734, 0.034451622515916824], [0.07949822396039963, 0.483990341424942, 0.023792315274477005, 0.35055384039878845], [0.2663685381412506, 0.4108807444572449, 0.3508042097091675, 0.31504857540130615], [0.16125237941741943, 0.21028673648834229, 0.4028345048427582, 0.3224439024925232], [0.2329816222190857, 0.38855111598968506, 0.4902689456939697, 0.017628690227866173], [0.28133541345596313, 0.292389452457428, 0.3052775263786316, 0.45975494384765625]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_916c3eac8830d2601aa865a08e76dbef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3411575257778168], [0.487751841545105], [0.0991593450307846], [0.005567505490034819], [0.4831579327583313], [0.3312598466873169], [0.3520636558532715], [0.29121044278144836], [0.12009581178426743]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.07136072963476181], [0.18888725340366364], [0.21746514737606049], [0.009641475975513458], [0.34512582421302795], [0.023270219564437866], [0.2941991984844208], [0.3037635385990143], [0.4153595566749573]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.3918236494064331], [0.17079667747020721], [0.2618381679058075], [0.10360092669725418], [0.2927932143211365], [0.150027334690094], [0.08460388332605362], [0.4861544370651245], [0.19172121584415436]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.18829287588596344], [0.34305354952812195], [0.08177930116653442], [0.3895333707332611], [0.46782585978507996], [0.33204638957977295], [0.23466411232948303], [0.13807879388332367], [0.03301508352160454]], dtype='float32').reshape([9, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_80df38937648711249bad4e8e17d1bdf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.35281503200531006], [0.18959030508995056], [0.45423388481140137], [0.15740129351615906], [0.15308204293251038], [0.25702497363090515], [0.2365979701280594], [0.4242403507232666], [0.4313344359397888]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.25974467396736145], [0.4788047671318054], [0.053342241793870926], [0.0840245708823204], [0.10440663248300552], [0.3625282645225525], [0.09382373839616776], [0.4737075865268707], [0.38082581758499146]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.13827703893184662], [0.1286333203315735], [0.22485984861850739], [0.15737760066986084], [0.2778044044971466], [0.15569062530994415], [0.15203706920146942], [0.4001633822917938], [0.4737466871738434]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.068621426820755], [0.3688132166862488], [0.14095377922058105], [0.1812393218278885], [0.4349217712879181], [0.4618573486804962], [0.2685330808162689], [0.38841864466667175], [0.36464715003967285]], dtype='float32').reshape([9, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_413450dd77fbb3dd8f359d47a48d483f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([5589, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5589, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5589, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5589, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_413450dd77fbb3dd8f359d47a48d483f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([5589, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5589, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5589, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([5589, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c7fd361eac45142d42fd4da185e70957(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.48915353417396545, 0.0599832721054554, 0.31578874588012695, 0.32014355063438416], [0.1614276021718979, 0.09353192150592804, 0.2042657434940338, 0.286808580160141], [0.4284842610359192, 0.20335653424263, 0.32612958550453186, 0.062386274337768555], [0.3643060624599457, 0.32442256808280945, 0.11638986319303513, 0.4475005269050598], [0.12722837924957275, 0.009367053396999836, 0.056048035621643066, 0.46019765734672546], [0.0880192220211029, 0.1442183554172516, 0.2839473783969879, 0.3002561628818512]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a961c7660b435186e4c58354b94b2ccb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.11336662620306015, 0.3186884820461273, 0.004312401637434959, 0.19526925683021545], [0.34463292360305786, 0.10320387780666351, 0.26624009013175964, 0.32023903727531433], [0.4841420650482178, 0.38910916447639465, 0.02423565462231636, 0.022353937849402428]], dtype='float32').reshape([3, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3aeb9f7ca1efbf7a4deb5fd98c19a015(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5dc53de0d2688a5b4748447c9855df4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.41314512491226196, 0.10803968459367752, 0.3724486231803894, 0.36033663153648376, 0.18769671022891998, 0.29229438304901123], dtype='float32').reshape([6]),
            paddle.to_tensor([0.10805502533912659, 0.01880381442606449, 0.22303974628448486, 0.19843772053718567, 0.027361908927559853, 0.16955791413784027], dtype='float32').reshape([6]),
            paddle.to_tensor([0.25822827219963074, 0.04662436991930008, 0.12753066420555115, 0.47527799010276794, 0.1672964096069336, 0.08656934648752213], dtype='float32').reshape([6]),
            paddle.to_tensor([0.08468656241893768, 0.306545615196228, 0.11176025122404099, 0.0717325508594513, 0.04547895863652229, 0.05067944526672363], dtype='float32').reshape([6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b141f1f0e296cf19c8c37efe8084fe4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5dc53de0d2688a5b4748447c9855df4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.029645506292581558, 0.004914592020213604, 0.09595445543527603, 0.39793166518211365, 0.4357508420944214, 0.4137418866157532], dtype='float32').reshape([6]),
            paddle.to_tensor([0.3976886570453644, 0.18534445762634277, 0.01889265514910221, 0.45858362317085266, 0.20388272404670715, 0.15198396146297455], dtype='float32').reshape([6]),
            paddle.to_tensor([0.23812781274318695, 0.19943611323833466, 0.06108584254980087, 0.09973878413438797, 0.3285718858242035, 0.48215657472610474], dtype='float32').reshape([6]),
            paddle.to_tensor([0.10711614042520523, 0.4060676693916321, 0.3443622589111328, 0.2741973400115967, 0.03332651033997536, 0.2714977264404297], dtype='float32').reshape([6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_893ada7398fbe1d8d5cf63d497a47072(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([1790, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1790, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1790, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1790, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_893ada7398fbe1d8d5cf63d497a47072(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([1790, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1790, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1790, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1790, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9d0bc4d30f5feec38ce72bc2cb3bf842(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.053108278661966324, 0.2642962634563446, 0.12306977063417435, 0.15607690811157227], [0.0833698958158493, 0.16468219459056854, 0.2770980894565582, 0.10666385293006897]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dad8dc9b552436e4339898c41a2b7248(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06575730443000793, 0.3563782870769501, 0.06684133410453796, 0.21661213040351868]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a0916d91e3810eb3aebe8614fcffe82e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([1512, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1512, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1512, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1512, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a0916d91e3810eb3aebe8614fcffe82e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([1512, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1512, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1512, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1512, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4876c8d847e7d98af0f409cd6aeddc40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.009746660478413105, 0.017361877486109734, 0.4770163595676422, 0.4977247714996338], [0.32846638560295105, 0.3449159860610962, 0.01744374819099903, 0.15449169278144836], [0.4994213879108429, 0.13463035225868225, 0.3727863132953644, 0.2907992899417877], [0.1041017398238182, 0.3148060441017151, 0.3109146058559418, 0.41507911682128906], [0.28564050793647766, 0.13475272059440613, 0.14990156888961792, 0.2785126864910126], [0.022030318155884743, 0.2485632300376892, 0.45357468724250793, 0.15473639965057373], [0.44435209035873413, 0.1585448682308197, 0.25553658604621887, 0.49046581983566284]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ec2e7144c2b044a62db2c435c4aa431d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.27415943145751953, 0.3107362985610962, 0.3970583975315094, 0.058175958693027496]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a721fb4675d5a97272d47f829b3988fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.30021312832832336]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.29566285014152527]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.4233570992946625]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.07360603660345078]], dtype='float32').reshape([1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_94bafbec21d4833dc21c3c503d21e745(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.17907217144966125]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.4305429756641388]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.11320116370916367]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.11482519656419754]], dtype='float32').reshape([1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b9a51fe9518fbe49d8b93dcbf8a96722(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.129848912358284], [0.02661607600748539], [0.27324140071868896], [0.39769837260246277], [0.01840292103588581], [0.15425024926662445]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.11104662716388702], [0.015465895645320415], [0.3915401101112366], [0.23462030291557312], [0.08867214620113373], [0.3940161466598511]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.03926978260278702], [0.08563093096017838], [0.3721535801887512], [0.18568728864192963], [0.22553320229053497], [0.4946292042732239]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.1343114972114563], [0.39124366641044617], [0.26737356185913086], [0.42793944478034973], [0.0809512734413147], [0.09805627912282944]], dtype='float32').reshape([6, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_98d1299bf6d9b436eb683671e36c106a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.36462271213531494], [0.15699154138565063], [0.009385272860527039], [0.49828392267227173], [0.1709592342376709], [0.18620306253433228]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.07934290170669556], [0.13667255640029907], [0.20768995583057404], [0.40495753288269043], [0.464950293302536], [0.4474965035915375]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.12608090043067932], [0.20497149229049683], [0.36897599697113037], [0.33328765630722046], [0.14978736639022827], [0.34729060530662537]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[0.07382135838270187], [0.4149003326892853], [0.2742626368999481], [0.411622017621994], [0.4005337357521057], [0.1369883418083191]], dtype='float32').reshape([6, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0e10b34533129dc08e01063fb3258777(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.12236736714839935, 0.023757170885801315, 0.2753857970237732, 0.0634797066450119], [0.3817211091518402, 0.3683093190193176, 0.09295232594013214, 0.3337242603302002], [0.1296713650226593, 0.17170575261116028, 0.23934559524059296, 0.2605445683002472], [0.33447957038879395, 0.19933679699897766, 0.28776758909225464, 0.13815784454345703], [0.4471212327480316, 0.07884528487920761, 0.03610456734895706, 0.03677848353981972]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d74e852b16e145c633c1c925c79dbece(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3657360374927521, 0.4700203537940979, 0.4082726836204529, 0.07482847571372986], [0.3315252363681793, 0.054085876792669296, 0.4313672184944153, 0.293525755405426], [0.27286601066589355, 0.18162350356578827, 0.05797296762466431, 0.47832754254341125], [0.07554564625024796, 0.04696071892976761, 0.2938121259212494, 0.4244130551815033], [0.2781350612640381, 0.035711824893951416, 0.16044847667217255, 0.07335866987705231], [0.126616969704628, 0.1599941849708557, 0.13273343443870544, 0.38682419061660767], [0.4670228362083435, 0.2216964066028595, 0.45244792103767395, 0.2261810600757599]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4f7653be174029c5ead87ef262043429(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4873305857181549, 0.3743271827697754, 0.16707535088062286, 0.4675159454345703], [0.007019971031695604, 0.4985446333885193, 0.3508499562740326, 0.226438507437706], [0.19130347669124603, 0.30717727541923523, 0.08884003758430481, 0.3845234215259552], [0.11493656039237976, 0.34949377179145813, 0.48343220353126526, 0.14999009668827057], [0.08428427577018738, 0.03870820626616478, 0.23276303708553314, 0.2538503110408783], [0.2606007158756256, 0.1668339967727661, 0.3094004988670349, 0.04491327702999115], [0.30882546305656433, 0.11893762648105621, 0.26801833510398865, 0.30436253547668457]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dddad4173698c7ba8d1b5da9e4a8c6b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4931090474128723, 0.16415029764175415, 0.1463773250579834, 0.001705510076135397]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0306db108e37d7164e038ff31d8fd99b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([2007, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2007, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2007, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2007, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0306db108e37d7164e038ff31d8fd99b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([2007, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2007, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2007, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2007, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f447239475201f1ca57d1c783a340133(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([4685, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4685, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4685, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4685, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f447239475201f1ca57d1c783a340133(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([4685, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4685, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4685, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4685, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_96ad60568fe92471c435a0529408551c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.08092506229877472, 0.4986841380596161, 0.3048730790615082, 0.2823871076107025], [0.4787973463535309, 0.11150021851062775, 0.2776354253292084, 0.2793109714984894], [0.2193784862756729, 0.03581467643380165, 0.47906818985939026, 0.38758349418640137], [0.2338825762271881, 0.25051364302635193, 0.43326351046562195, 0.22051788866519928], [0.1832806020975113, 0.28888657689094543, 0.3417982757091522, 0.382455974817276]], dtype='float32').reshape([5, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cebd14b8f5a7fd770b58b54883e5c6ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([1050, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1050, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1050, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1050, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cebd14b8f5a7fd770b58b54883e5c6ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([1050, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1050, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1050, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1050, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fa9b0901c83c9917e66ffe8b7754a21d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.323615163564682], [0.30178794264793396], [0.1656930297613144], [0.11601903289556503], [0.10725655406713486]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.2432054579257965], [0.38019073009490967], [0.11684619635343552], [0.4634474217891693], [0.16990448534488678]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.20664435625076294], [0.4538969397544861], [0.20020511746406555], [0.4132084846496582], [0.2432897984981537]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.17348651587963104], [0.32497742772102356], [0.003286341205239296], [0.16559453308582306], [0.24491815268993378]], dtype='float32').reshape([5, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1ecdb9e8b1d591a6ed2a2e0b67008768(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.13989704847335815], [0.4314858913421631], [0.14878864586353302], [0.40420448780059814], [0.3773234188556671]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.3156962990760803], [0.09945223480463028], [0.2634713351726532], [0.11271272599697113], [0.23488155007362366]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.004550704266875982], [0.4207856059074402], [0.24096810817718506], [0.004030225798487663], [0.03893037885427475]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.36014536023139954], [0.19286362826824188], [0.07153784483671188], [0.069424569606781], [0.2985104024410248]], dtype='float32').reshape([5, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c69168bdf0f9cec09185265cfd9b4008(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.2901870906352997, 0.0027597020380198956, 0.098127081990242, 0.32107290625572205]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5a090f00639612906ca88e8e9a87913f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.15391285717487335, 0.12124611437320709, 0.10962545871734619, 0.4960550367832184]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d5c59601db3803c64fa3ee0225fa74f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([2394, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2394, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2394, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2394, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d5c59601db3803c64fa3ee0225fa74f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([2394, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2394, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2394, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2394, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c53987450ddcee0b601c9e680dc1798e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c53987450ddcee0b601c9e680dc1798e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d4edba6b4182fdf49375078cb84a88e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([3758, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3758, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3758, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3758, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d4edba6b4182fdf49375078cb84a88e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([3758, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3758, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3758, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([3758, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_230e94c2795e2504963e7d8492f59621(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06738543510437012, 0.09491246193647385, 0.13308154046535492, 0.33248892426490784]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_947a3f1c012f98812b14ab24bde9d691(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.15036208927631378], [0.1331283152103424], [0.2937854826450348], [0.4619079828262329]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.1913178265094757], [0.294893741607666], [0.036232516169548035], [0.43063637614250183]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.06513538211584091], [0.3193783462047577], [0.09922898560762405], [0.1432037353515625]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.17245835065841675], [0.15862299501895905], [0.4525022804737091], [0.1308891922235489]], dtype='float32').reshape([4, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9fb7ed1699e3b4f5e26c5ed6884704cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.33588987588882446], [0.2792902886867523], [0.1107465997338295], [0.3460559844970703]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.2022257000207901], [0.14955280721187592], [0.286821573972702], [0.023547960445284843]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.1513662487268448], [0.28769925236701965], [0.18229663372039795], [0.12690480053424835]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.25924187898635864], [0.346165269613266], [0.1456291526556015], [0.23360179364681244]], dtype='float32').reshape([4, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a056baa9f77be60e9a771adeaf2f2e70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([2008, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2008, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2008, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2008, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a056baa9f77be60e9a771adeaf2f2e70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([2008, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2008, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2008, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([2008, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8b7dd1dfd611cb0fb037216432d1ab85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([4295, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4295, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4295, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4295, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8b7dd1dfd611cb0fb037216432d1ab85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.uniform([4295, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4295, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4295, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([4295, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fb83f3d904467a9cad023fe66f73833b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4620692729949951, 0.10149343311786652, 0.038771647959947586, 0.23243044316768646], [0.15344084799289703, 0.06268923729658127, 0.34847208857536316, 0.42999449372291565], [0.15190163254737854, 0.4104097783565521, 0.365616112947464, 0.1296401470899582], [0.27523496747016907, 0.29393917322158813, 0.0716368556022644, 0.49047335982322693], [0.012108967639505863, 0.4853935241699219, 0.2429102063179016, 0.41299933195114136], [0.030071891844272614, 0.3988513946533203, 0.48055341839790344, 0.041261013597249985], [0.1597491055727005, 0.31782782077789307, 0.05290887504816055, 0.13929510116577148]], dtype='float32').reshape([7, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_43019152c736721518f41720e3eaf3d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e734ca0b10721d7d688a261329cb981a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3801068067550659, 0.3316466212272644, 0.17720377445220947, 0.15354834496974945], [0.40502336621284485, 0.17050214111804962, 0.27920740842819214, 0.3615216910839081], [0.03935656696557999, 0.3974427878856659, 0.4215141534805298, 0.31277039647102356], [0.23049986362457275, 0.42653417587280273, 0.13485263288021088, 0.48253151774406433], [0.4592631459236145, 0.49990564584732056, 0.479204922914505, 0.3003561496734619], [0.4457668662071228, 0.31206563115119934, 0.4506196677684784, 0.040063779801130295]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            paddle.to_tensor([], dtype='float32').reshape([0, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
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
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()



if __name__ == '__main__':
    unittest.main()