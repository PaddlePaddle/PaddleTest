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
class PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8af2b94ad69e503cb9b248227a80e28d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8bc148db5d1868af7ef15bf4c7209676(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_6869b440549846553242dab458ee42b6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6d32391b3e11daa3fe574568322a5564(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6869b440549846553242dab458ee42b6
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.24821408092975616]]], dtype='float32').reshape([1, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_25d97c980116915b97d7f1dd8630de8d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[12096, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[12096, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_77c5d87ca6d7e40a8ca31b0c9c1912e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25d97c980116915b97d7f1dd8630de8d
    def get_inputs(self):
        return [
            paddle.uniform([12096, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([12096, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7f2410859a0475f8ba1e59ebe88a3107(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_926863dabbe694f2c349b4c63553e89c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_603d97710ed6a19cc34142e06ccee4c9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_43663605b8b323d798e72d991a66570b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_603d97710ed6a19cc34142e06ccee4c9
    def get_inputs(self):
        return [
            paddle.to_tensor([1100.56396484375], dtype='float32').reshape([1]),
            paddle.to_tensor(8732.0, dtype='float32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_6291b528d04c1a686a627623cddccfb6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_108791a8625df034f927e15d4be77d0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6291b528d04c1a686a627623cddccfb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.0013434607535600662], [1.6916560525714885e-06], [8.434222763753496e-06], [0.004283439368009567], [0.000182949414011091], [0.005898493807762861]]], dtype='float32').reshape([1, 6, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_086f05b616ee85c1291331038d540688(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6291b528d04c1a686a627623cddccfb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.016174612566828728], [0.0008331138524226844], [0.00025323880254291], [0.0029187118634581566], [0.0036240555346012115], [0.016160449013113976]]], dtype='float32').reshape([1, 6, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_94e22fb993cbbb7af3d3ca9386fc9727(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6869b440549846553242dab458ee42b6
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.12284519523382187], [0.14520911872386932], [0.1053224429488182], [0.22626642882823944], [0.03763167932629585], [0.22110427916049957]]], dtype='float32').reshape([1, 6, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_545d921ce54972722f299efa19089ae2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2efe02dd038765fcf805744686e8bba8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8af2b94ad69e503cb9b248227a80e28d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_128f67565db93e81675dd337049efee6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3cc383d532fb1ba76352e619d107dd00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(10.082260131835938, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1119c15e6d68e4b1ac139a291e88d09c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(2.158231735229492, dtype='float32').reshape([]),
            paddle.to_tensor([2.0], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_df13a428be372fa1e867c6ad008050a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_d038adfdf55cd8f949a8fe4193c0f2c2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d1271772357086f822c812e9a0022940(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d038adfdf55cd8f949a8fe4193c0f2c2
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_d1271772357086f822c812e9a0022940(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d038adfdf55cd8f949a8fe4193c0f2c2
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_c1fd528d825a0db3ae914a65267aff50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(64224.71875, dtype='float32').reshape([]),
            paddle.to_tensor([0.3421415686607361], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_020301c8992d8870b2d762fe119c80f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(98870.71875, dtype='float32').reshape([]),
            paddle.to_tensor([0.3421415686607361], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7a856781ea18ec2fe9b535d12d34d50b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(949.3621826171875, dtype='float32').reshape([]),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c6bf680b69490fe029e977c78d66dc6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c8eb03c9b06c9642e62120859f5de760(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_60f5a98ea45ebadb60b7305290c14e0a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[5376, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[5376, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_600dc942a0e746b736df49ab662fa096(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60f5a98ea45ebadb60b7305290c14e0a
    def get_inputs(self):
        return [
            paddle.uniform([5376, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([5376, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4850c1e2e60ea7a945cff314249a9a4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e1651d47886b31cc4d74a01c4a69a08e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8f22295d2bb3bb329674a0a6b5210376(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[-0.08620942384004593], [-0.16619189083576202], [-0.08307240903377533], [0.014591867104172707], [0.08238504827022552], [-0.01507766917347908], [0.04899847134947777], [0.04062556102871895], [-0.060701437294483185]], dtype='float32').reshape([9, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4e297414be34144b0a9bdbb2568862f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.09824222326278687], [0.22777122259140015], [0.07219719141721725], [-0.008609030395746231], [-0.018118947744369507], [0.01921701990067959], [-0.03839242085814476], [-0.08942095935344696], [0.03099716641008854]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.012032799422740936], [0.061579324305057526], [-0.0108752166852355], [0.0059828367084264755], [0.06426610052585602], [0.004139351192861795], [0.010606050491333008], [-0.048795394599437714], [-0.029704270884394646]], dtype='float32').reshape([9, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_ecb18185dc865f9a1072115911e91a52(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c4379464ef629ca62d03233d1fb0d854(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ecb18185dc865f9a1072115911e91a52
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.042378272861242294], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fa2f0dc377e994bd38cee2181ad1b672(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d038adfdf55cd8f949a8fe4193c0f2c2
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_fa2f0dc377e994bd38cee2181ad1b672(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d038adfdf55cd8f949a8fe4193c0f2c2
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_6a849b3a1e07c067b9877634cfbeab5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(47500.5703125, dtype='float32').reshape([]),
            paddle.to_tensor([0.04790016636252403], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_26d6b8b4c1fdc3f21262436a1739bc9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(3961.21923828125, dtype='float32').reshape([]),
            paddle.to_tensor([0.04790016636252403], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_5e8106aa512a8da9fae679f4b94ae655(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_83fff9c5f1e5b0ed9b7020fd1ccfa265(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e8106aa512a8da9fae679f4b94ae655
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, -0.0, -0.0, 0.0, 0.0, 0.001345331547781825], dtype='float32').reshape([6]),
            paddle.to_tensor([-9.102913463721052e-05, 0.026912759989500046, -0.025511933490633965, -0.01090479176491499, -0.004545001778751612, 0.08324715495109558], dtype='float32').reshape([6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fbdc772c0971555e6dc1c5950f50c685(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e8106aa512a8da9fae679f4b94ae655
    def get_inputs(self):
        return [
            paddle.to_tensor([0.017981061711907387, 0.0784120112657547, 0.026303216814994812, 0.03572332113981247, 0.04680575430393219, 0.029992714524269104], dtype='float32').reshape([6]),
            paddle.to_tensor([0.11923456937074661, 0.1395626664161682, 0.05190751701593399, 0.1680527925491333, 0.01911763660609722, 0.19781967997550964], dtype='float32').reshape([6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7051583dbcd37f14bbe53d6a13ad97a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e8106aa512a8da9fae679f4b94ae655
    def get_inputs(self):
        return [
            paddle.to_tensor([0.08425328135490417, 0.2561700940132141, -0.28105923533439636, -0.29649215936660767, -0.22445550560951233, 0.033966973423957825], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.0010804235935211182, 0.10505816340446472, 0.0907706618309021, 0.036779358983039856, 0.020249009132385254, 0.08722567558288574], dtype='float32').reshape([6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2adc2778f8f7f57aaf8176aa971def83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e8106aa512a8da9fae679f4b94ae655
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.45551609992980957, 0.09419456124305725, -0.12396994233131409, 0.35419464111328125, 0.0938267707824707, 0.33792948722839355], dtype='float32').reshape([6]),
            paddle.to_tensor([0.317158967256546, -0.1187971979379654, 0.21494784951210022, -0.008993566036224365, -0.1928962767124176, 0.24155835807323456], dtype='float32').reshape([6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4741ba6c1ee28cb6f019ef6de85a6506(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e8106aa512a8da9fae679f4b94ae655
    def get_inputs(self):
        return [
            paddle.to_tensor([0.14366944134235382, 1.390091896057129, 0.2191043496131897, 0.0038948804140090942, 0.4283895790576935, 0.13579584658145905], dtype='float32').reshape([6]),
            paddle.to_tensor([1.1436694860458374, 2.390091896057129, 1.219104290008545, 1.0038949251174927, 1.428389549255371, 1.1196352243423462], dtype='float32').reshape([6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4447fb7dae1296032a17c359e3607109(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d038adfdf55cd8f949a8fe4193c0f2c2
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_4447fb7dae1296032a17c359e3607109(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d038adfdf55cd8f949a8fe4193c0f2c2
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_449d7ac86472fa429052fabc3c12d6aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(-2045479.5, dtype='float32').reshape([]),
            paddle.to_tensor([0.05314517021179199], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c91bff3f85bce9f0c3ef918ea43ebc4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(102253.9921875, dtype='float32').reshape([]),
            paddle.to_tensor([0.05314517021179199], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0ea36eb72ccdff63e8c6b71cea47297c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(935.5975341796875, dtype='float32').reshape([]),
            paddle.to_tensor([4.0], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_3df0b87d8aa6daf297efe06a6e629ab3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[8400, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[8400, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_04093335fce749ec1a0e1a09ab326edd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3df0b87d8aa6daf297efe06a6e629ab3
    def get_inputs(self):
        return [
            paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([8400, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_bbdc3ff9f0a348b18d59d65d7a016467(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_404e76c972f50620d043d72d23b27502(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbdc3ff9f0a348b18d59d65d7a016467
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 38, 38], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_926863dabbe694f2c349b4c63553e89c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c8eb03c9b06c9642e62120859f5de760(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a0cf26bc8f0b10e0a50e23898c94c485(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d038adfdf55cd8f949a8fe4193c0f2c2
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_a0cf26bc8f0b10e0a50e23898c94c485(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d038adfdf55cd8f949a8fe4193c0f2c2
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_5c1afaeeadf810bf90bf206b0bbfe30f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(535613.5625, dtype='float32').reshape([]),
            paddle.to_tensor([0.08735578507184982], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_93cc39d080b3a3c61d27eed1bbfe7e35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(83143.28125, dtype='float32').reshape([]),
            paddle.to_tensor([0.08735578507184982], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_04638c6c9ef8bec790089118b47183a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6869b440549846553242dab458ee42b6
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.2471751719713211], [0.2486463189125061]]], dtype='float32').reshape([1, 2, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2efe02dd038765fcf805744686e8bba8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8bc148db5d1868af7ef15bf4c7209676(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_00161da141651d14ee93284505e58b77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.03623388707637787]], dtype='float32').reshape([1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9e68d20a54661d6405353153d9a676de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.09822024405002594]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.06198635324835777]], dtype='float32').reshape([1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e75d734edbb2f67db3c3a76e070ada43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.10920996963977814], [-0.007211522199213505], [0.011571147479116917], [-0.15353214740753174], [-0.018995165824890137], [0.04299311712384224]], dtype='float32').reshape([6, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_aca599ce113d41fcc2f8210dfc8110fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.057135917246341705], [0.052032552659511566], [0.011957314796745777], [0.1162772849202156], [0.020450709387660027], [0.01571604236960411]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.05207405239343643], [0.044821031391620636], [0.023528462275862694], [-0.03725486248731613], [0.0014555440284311771], [0.05870915949344635]], dtype='float32').reshape([6, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4614224f5e09867a8a73cf22f34134d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6869b440549846553242dab458ee42b6
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.24266885221004486]]], dtype='float32').reshape([1, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_1252e93a0fedea11d028fadaaa588f5e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b28f7def5aa4b7ff00ac21ea08192327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1252e93a0fedea11d028fadaaa588f5e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c6bf680b69490fe029e977c78d66dc6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7f2410859a0475f8ba1e59ebe88a3107(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2d8964b82b81f42d9b07c7be270e5e3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(59.66321563720703, dtype='float32').reshape([]),
            paddle.to_tensor([7.0], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f78ce61216bef93add44e0a24746f675(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(548.6436157226562, dtype='float32').reshape([]),
            paddle.to_tensor([4.0], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d40de3914c99fed651e8f344ba0818d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d038adfdf55cd8f949a8fe4193c0f2c2
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_d40de3914c99fed651e8f344ba0818d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d038adfdf55cd8f949a8fe4193c0f2c2
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_bb89190f7a487956a4185509b9ccc802(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(176725.484375, dtype='float32').reshape([]),
            paddle.to_tensor([0.42551374435424805], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a06e3e5726019a9383974c445ed82d41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(116908.9140625, dtype='float32').reshape([]),
            paddle.to_tensor([0.42551374435424805], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_72c6f96f4bb201ca2238f704ef7b4447(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1252e93a0fedea11d028fadaaa588f5e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 152, 272], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_65b0c45d8679112e282521cae5d40f45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c74958faf68504538c189066024cd619(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ecb18185dc865f9a1072115911e91a52
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3039810359477997], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_48fbe7651a05653b60bce51bb86c85d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d038adfdf55cd8f949a8fe4193c0f2c2
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_48fbe7651a05653b60bce51bb86c85d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d038adfdf55cd8f949a8fe4193c0f2c2
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_82746842aee34a648d162a4993bc5dfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(52401.5390625, dtype='float32').reshape([]),
            paddle.to_tensor([0.27586016058921814], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9c37bc032b078c9961fdd492e848830a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(268655.5, dtype='float32').reshape([]),
            paddle.to_tensor([0.27586016058921814], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_81a9432f12e1c1e6a3eb2311ff170c13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_603d97710ed6a19cc34142e06ccee4c9
    def get_inputs(self):
        return [
            paddle.to_tensor([306.7394104003906], dtype='float32').reshape([1]),
            paddle.to_tensor(2434.0, dtype='float32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_afe16048c9c27017606b3120f6aa924a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d038adfdf55cd8f949a8fe4193c0f2c2
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_afe16048c9c27017606b3120f6aa924a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d038adfdf55cd8f949a8fe4193c0f2c2
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_27894364c5eef25131335a2ff1243d34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(33165.703125, dtype='float32').reshape([]),
            paddle.to_tensor([0.3705717623233795], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eae73041ced26629de998ba339483e37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(15098.0625, dtype='float32').reshape([]),
            paddle.to_tensor([0.3705717623233795], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_545d921ce54972722f299efa19089ae2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_feebc6be97f13502b438ab14fe6e4a58(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[100, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ee1910fdb1d3b18c9af00c2b1ec3685e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_feebc6be97f13502b438ab14fe6e4a58
    def get_inputs(self):
        return [
            paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a591aaef7f5333704424b228d7fd7318(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1558956503868103, 0.27804291248321533, 0.38011226058006287, 0.020000966265797615], [0.3080019950866699, 0.016835307702422142, 0.18333855271339417, 0.12878437340259552]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([[0.0657414123415947, 0.2821248769760132, 0.41800209879875183, 0.2077385038137436], [0.06425489485263824, 0.08323095738887787, 0.13659104704856873, 0.3990457355976105]], dtype='float32').reshape([2, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4850c1e2e60ea7a945cff314249a9a4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_f3be92c408228e6de64cd519300442cd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6069, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[6069, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b1b86298853875c59536bbb5bc0ac6f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3be92c408228e6de64cd519300442cd
    def get_inputs(self):
        return [
            paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([6069, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_672487b7a61e3971e7c1d3df47671b15(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[300, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_aeec1bb5bdb11ad5988b9587b8fe1444(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_672487b7a61e3971e7c1d3df47671b15
    def get_inputs(self):
        return [
            paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a7e08227a125f3692f04448985f4022e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4007115662097931, 0.12639737129211426, 0.4112824499607086, 0.008094380609691143], [0.29197075963020325, 0.24120639264583588, 0.499051958322525, 0.09907882660627365]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([[0.27978193759918213, 0.12155638635158539, 0.46089935302734375, 0.44037988781929016], [0.11692929267883301, 0.10377798974514008, 0.16839845478534698, 0.04607399180531502]], dtype='float32').reshape([2, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_31d991ddc31e6d08482bcc84c03b3344(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.01218876801431179], [-0.007885221391916275], [-0.07408457249403], [-0.14463911950588226], [-0.017956875264644623]], dtype='float32').reshape([5, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b5f3283f3f267a3a7c358cff3b14693c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.03584502637386322], [0.041892047971487045], [0.04215621203184128], [0.11972222477197647], [-0.01645669713616371]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.02365626022219658], [0.03400682657957077], [-0.03192836046218872], [-0.02491689659655094], [-0.034413572400808334]], dtype='float32').reshape([5, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_478a807de85e8d96e4e107a46ef05628(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ecb18185dc865f9a1072115911e91a52
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3863242268562317], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_df13a428be372fa1e867c6ad008050a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e1651d47886b31cc4d74a01c4a69a08e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d558c26136203421c57b4b72437d23ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0b9284a8819be45854213b04beead98d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d038adfdf55cd8f949a8fe4193c0f2c2
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_0b9284a8819be45854213b04beead98d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d038adfdf55cd8f949a8fe4193c0f2c2
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_7a63521ad0c59c53e2e94d5755d30e5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(-260738.65625, dtype='float32').reshape([]),
            paddle.to_tensor([0.3184764087200165], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_14a1aa5deb1d431218b84f1149f3dd62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(133816.90625, dtype='float32').reshape([]),
            paddle.to_tensor([0.3184764087200165], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a1c8e4eb814d3d6329f1a8471d91ec9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d038adfdf55cd8f949a8fe4193c0f2c2
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_a1c8e4eb814d3d6329f1a8471d91ec9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d038adfdf55cd8f949a8fe4193c0f2c2
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_d025a29fac2c0205b7c07d9b388e9828(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(75233.65625, dtype='float32').reshape([]),
            paddle.to_tensor([0.3707900643348694], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2998351683913c56cf106a679de471b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(170275.46875, dtype='float32').reshape([]),
            paddle.to_tensor([0.3707900643348694], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ee19c9b186af3c1229c34808eda97067(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d038adfdf55cd8f949a8fe4193c0f2c2
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_ee19c9b186af3c1229c34808eda97067(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d038adfdf55cd8f949a8fe4193c0f2c2
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_cdac2e0eb19d351224fa508b7173bb28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(-657314.5, dtype='float32').reshape([]),
            paddle.to_tensor([0.4422951340675354], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_643e8d6cd5d76dc7a16d2f080e280855(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(219193.84375, dtype='float32').reshape([]),
            paddle.to_tensor([0.4422951340675354], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3fba79b519a7cd10615645ebef50d98a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ecb18185dc865f9a1072115911e91a52
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2676987051963806], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_65b0c45d8679112e282521cae5d40f45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8ebec33baf768fee56f0994f294de81d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(15.995766639709473, dtype='float32').reshape([]),
            paddle.to_tensor([3.0], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_47d5926b1613ce642fc4550e8e7db757(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20267, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[20267, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f7ece3b82da49f3f7b38af7f85c6d03e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47d5926b1613ce642fc4550e8e7db757
    def get_inputs(self):
        return [
            paddle.uniform([20267, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([20267, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2a315fdbd49fc237ba883243237d21fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.054475411772727966], [0.008385568857192993], [0.03203628212213516], [-0.04012038931250572]], dtype='float32').reshape([4, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2aa6c5b73ad00089c1965f5e7df71d9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.12263159453868866], [0.014178849756717682], [-0.023235611617565155], [0.08588115870952606]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.0681561827659607], [0.022564418613910675], [0.008800670504570007], [0.04576076939702034]], dtype='float32').reshape([4, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7d89ade18fd38022e228b2101de3f481(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(3.7783987522125244, dtype='float32').reshape([]),
            paddle.to_tensor([7.0], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d621922a8656947b31edc2fe08edc965(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d038adfdf55cd8f949a8fe4193c0f2c2
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_d621922a8656947b31edc2fe08edc965(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d038adfdf55cd8f949a8fe4193c0f2c2
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_ad693fe09381b3c8f6c2447a39984006(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(17242.330078125, dtype='float32').reshape([]),
            paddle.to_tensor([0.4598577618598938], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3eaec148fec9050007f5a0303472795e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(29657.19921875, dtype='float32').reshape([]),
            paddle.to_tensor([0.4598577618598938], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dc47716b49be04ed7fbd3cc26c111f7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ecb18185dc865f9a1072115911e91a52
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.27220335602760315], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d5459f296c9b1a161fbad0628977fdf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(34.268775939941406, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_543225686aaffea8a323086251667a86(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[6804, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[6804, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1bd0e6845022d3deae5d66ed1e8d4b23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_543225686aaffea8a323086251667a86
    def get_inputs(self):
        return [
            paddle.uniform([6804, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([6804, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d558c26136203421c57b4b72437d23ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0b3678cf7736712aa05c8023c6b53cd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(234.89334106445312, dtype='float32').reshape([]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f15ff585951d7860623a9f32f80d8ee5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(137.79554748535156, dtype='float32').reshape([]),
            paddle.to_tensor([7.0], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_128f67565db93e81675dd337049efee6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_73ad1292c5bdfe6e4908735e0fa958d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1e6c6f99a8c2b95daeea57be9775efc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d038adfdf55cd8f949a8fe4193c0f2c2
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_1e6c6f99a8c2b95daeea57be9775efc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d038adfdf55cd8f949a8fe4193c0f2c2
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_785269ed177d3d517650f00a54b11016(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(594974.5625, dtype='float32').reshape([]),
            paddle.to_tensor([0.39719903469085693], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_02d1080c8e5e9bcb4b615436b788ad05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(236797.140625, dtype='float32').reshape([]),
            paddle.to_tensor([0.39719903469085693], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_73ad1292c5bdfe6e4908735e0fa958d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8af2b94ad69e503cb9b248227a80e28d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8bc148db5d1868af7ef15bf4c7209676(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4713fecdfb171cbf657662484b95a4fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6291b528d04c1a686a627623cddccfb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.24821408092975616]]], dtype='float32').reshape([1, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6462ec02d8d2f3f158e978ae530ed59b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
            paddle.uniform([12096, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([12096, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7f2410859a0475f8ba1e59ebe88a3107(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_926863dabbe694f2c349b4c63553e89c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_43663605b8b323d798e72d991a66570b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_603d97710ed6a19cc34142e06ccee4c9
    def get_inputs(self):
        return [
            paddle.to_tensor([1100.56396484375], dtype='float32').reshape([1]),
            paddle.to_tensor(8732.0, dtype='float32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_108791a8625df034f927e15d4be77d0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6291b528d04c1a686a627623cddccfb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.0013434607535600662], [1.6916560525714885e-06], [8.434222763753496e-06], [0.004283439368009567], [0.000182949414011091], [0.005898493807762861]]], dtype='float32').reshape([1, 6, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_086f05b616ee85c1291331038d540688(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6291b528d04c1a686a627623cddccfb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.016174612566828728], [0.0008331138524226844], [0.00025323880254291], [0.0029187118634581566], [0.0036240555346012115], [0.016160449013113976]]], dtype='float32').reshape([1, 6, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3d8dd34fa6ba410457317039b123ed03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6291b528d04c1a686a627623cddccfb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.12284519523382187], [0.14520911872386932], [0.1053224429488182], [0.22626642882823944], [0.03763167932629585], [0.22110427916049957]]], dtype='float32').reshape([1, 6, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_545d921ce54972722f299efa19089ae2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2efe02dd038765fcf805744686e8bba8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8af2b94ad69e503cb9b248227a80e28d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_128f67565db93e81675dd337049efee6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3cc383d532fb1ba76352e619d107dd00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(10.082260131835938, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1119c15e6d68e4b1ac139a291e88d09c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(2.158231735229492, dtype='float32').reshape([]),
            paddle.to_tensor([2.0], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_df13a428be372fa1e867c6ad008050a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_91bf509dab554c1b9a831f25c552380b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_91bf509dab554c1b9a831f25c552380b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_c1fd528d825a0db3ae914a65267aff50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(64224.71875, dtype='float32').reshape([]),
            paddle.to_tensor([0.3421415686607361], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_020301c8992d8870b2d762fe119c80f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(98870.71875, dtype='float32').reshape([]),
            paddle.to_tensor([0.3421415686607361], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7a856781ea18ec2fe9b535d12d34d50b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(949.3621826171875, dtype='float32').reshape([]),
            paddle.to_tensor([8.0], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c6bf680b69490fe029e977c78d66dc6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c8eb03c9b06c9642e62120859f5de760(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_af7c301f2d77ad067ca5b0651cc985e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
            paddle.uniform([5376, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([5376, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4850c1e2e60ea7a945cff314249a9a4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e1651d47886b31cc4d74a01c4a69a08e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8f22295d2bb3bb329674a0a6b5210376(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[-0.08620942384004593], [-0.16619189083576202], [-0.08307240903377533], [0.014591867104172707], [0.08238504827022552], [-0.01507766917347908], [0.04899847134947777], [0.04062556102871895], [-0.060701437294483185]], dtype='float32').reshape([9, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4e297414be34144b0a9bdbb2568862f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.09824222326278687], [0.22777122259140015], [0.07219719141721725], [-0.008609030395746231], [-0.018118947744369507], [0.01921701990067959], [-0.03839242085814476], [-0.08942095935344696], [0.03099716641008854]], dtype='float32').reshape([9, 1]),
            paddle.to_tensor([[0.012032799422740936], [0.061579324305057526], [-0.0108752166852355], [0.0059828367084264755], [0.06426610052585602], [0.004139351192861795], [0.010606050491333008], [-0.048795394599437714], [-0.029704270884394646]], dtype='float32').reshape([9, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_11a6c2ce7213d45a7b8f7fce3e64d8a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2bbc3689ae45050bcd755fceb120b1ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11a6c2ce7213d45a7b8f7fce3e64d8a6
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.042378272861242294], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fdee672a1ce83d8f901d10de3aa46e99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_fdee672a1ce83d8f901d10de3aa46e99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_6a849b3a1e07c067b9877634cfbeab5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(47500.5703125, dtype='float32').reshape([]),
            paddle.to_tensor([0.04790016636252403], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_26d6b8b4c1fdc3f21262436a1739bc9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(3961.21923828125, dtype='float32').reshape([]),
            paddle.to_tensor([0.04790016636252403], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_83fff9c5f1e5b0ed9b7020fd1ccfa265(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e8106aa512a8da9fae679f4b94ae655
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, -0.0, -0.0, 0.0, 0.0, 0.001345331547781825], dtype='float32').reshape([6]),
            paddle.to_tensor([-9.102913463721052e-05, 0.026912759989500046, -0.025511933490633965, -0.01090479176491499, -0.004545001778751612, 0.08324715495109558], dtype='float32').reshape([6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fbdc772c0971555e6dc1c5950f50c685(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e8106aa512a8da9fae679f4b94ae655
    def get_inputs(self):
        return [
            paddle.to_tensor([0.017981061711907387, 0.0784120112657547, 0.026303216814994812, 0.03572332113981247, 0.04680575430393219, 0.029992714524269104], dtype='float32').reshape([6]),
            paddle.to_tensor([0.11923456937074661, 0.1395626664161682, 0.05190751701593399, 0.1680527925491333, 0.01911763660609722, 0.19781967997550964], dtype='float32').reshape([6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7051583dbcd37f14bbe53d6a13ad97a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e8106aa512a8da9fae679f4b94ae655
    def get_inputs(self):
        return [
            paddle.to_tensor([0.08425328135490417, 0.2561700940132141, -0.28105923533439636, -0.29649215936660767, -0.22445550560951233, 0.033966973423957825], dtype='float32').reshape([6]),
            paddle.to_tensor([-0.0010804235935211182, 0.10505816340446472, 0.0907706618309021, 0.036779358983039856, 0.020249009132385254, 0.08722567558288574], dtype='float32').reshape([6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2adc2778f8f7f57aaf8176aa971def83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e8106aa512a8da9fae679f4b94ae655
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.45551609992980957, 0.09419456124305725, -0.12396994233131409, 0.35419464111328125, 0.0938267707824707, 0.33792948722839355], dtype='float32').reshape([6]),
            paddle.to_tensor([0.317158967256546, -0.1187971979379654, 0.21494784951210022, -0.008993566036224365, -0.1928962767124176, 0.24155835807323456], dtype='float32').reshape([6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4741ba6c1ee28cb6f019ef6de85a6506(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e8106aa512a8da9fae679f4b94ae655
    def get_inputs(self):
        return [
            paddle.to_tensor([0.14366944134235382, 1.390091896057129, 0.2191043496131897, 0.0038948804140090942, 0.4283895790576935, 0.13579584658145905], dtype='float32').reshape([6]),
            paddle.to_tensor([1.1436694860458374, 2.390091896057129, 1.219104290008545, 1.0038949251174927, 1.428389549255371, 1.1196352243423462], dtype='float32').reshape([6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c0ab5c28d4b85d584c7d0f500b53e3e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_c0ab5c28d4b85d584c7d0f500b53e3e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_449d7ac86472fa429052fabc3c12d6aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(-2045479.5, dtype='float32').reshape([]),
            paddle.to_tensor([0.05314517021179199], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c91bff3f85bce9f0c3ef918ea43ebc4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(102253.9921875, dtype='float32').reshape([]),
            paddle.to_tensor([0.05314517021179199], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0ea36eb72ccdff63e8c6b71cea47297c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(935.5975341796875, dtype='float32').reshape([]),
            paddle.to_tensor([4.0], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fb02871ddcb4313c0b1c90f660858193(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
            paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([8400, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_67d7f8cfed71c4d3601fdbefae2a90db(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 / input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d5a7f87e8241495cb9482d5fec4d9830(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67d7f8cfed71c4d3601fdbefae2a90db
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 38, 38], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_926863dabbe694f2c349b4c63553e89c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c8eb03c9b06c9642e62120859f5de760(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f71dac9f2930391f86a9eb17f39f4f13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_f71dac9f2930391f86a9eb17f39f4f13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_5c1afaeeadf810bf90bf206b0bbfe30f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(535613.5625, dtype='float32').reshape([]),
            paddle.to_tensor([0.08735578507184982], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_93cc39d080b3a3c61d27eed1bbfe7e35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(83143.28125, dtype='float32').reshape([]),
            paddle.to_tensor([0.08735578507184982], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_91356e274ff355e16638a87d91f634fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6291b528d04c1a686a627623cddccfb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.2471751719713211], [0.2486463189125061]]], dtype='float32').reshape([1, 2, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2efe02dd038765fcf805744686e8bba8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8bc148db5d1868af7ef15bf4c7209676(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_00161da141651d14ee93284505e58b77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[-0.03623388707637787]], dtype='float32').reshape([1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9e68d20a54661d6405353153d9a676de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.09822024405002594]], dtype='float32').reshape([1, 1]),
            paddle.to_tensor([[0.06198635324835777]], dtype='float32').reshape([1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e75d734edbb2f67db3c3a76e070ada43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.10920996963977814], [-0.007211522199213505], [0.011571147479116917], [-0.15353214740753174], [-0.018995165824890137], [0.04299311712384224]], dtype='float32').reshape([6, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_aca599ce113d41fcc2f8210dfc8110fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.057135917246341705], [0.052032552659511566], [0.011957314796745777], [0.1162772849202156], [0.020450709387660027], [0.01571604236960411]], dtype='float32').reshape([6, 1]),
            paddle.to_tensor([[-0.05207405239343643], [0.044821031391620636], [0.023528462275862694], [-0.03725486248731613], [0.0014555440284311771], [0.05870915949344635]], dtype='float32').reshape([6, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4fb3428a5d8851dcf967d82b5911d927(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6291b528d04c1a686a627623cddccfb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[0.24266885221004486]]], dtype='float32').reshape([1, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c68261dbce3e9f585b6d870b3c7076ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67d7f8cfed71c4d3601fdbefae2a90db
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 34], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c6bf680b69490fe029e977c78d66dc6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7f2410859a0475f8ba1e59ebe88a3107(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2d8964b82b81f42d9b07c7be270e5e3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(59.66321563720703, dtype='float32').reshape([]),
            paddle.to_tensor([7.0], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f78ce61216bef93add44e0a24746f675(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(548.6436157226562, dtype='float32').reshape([]),
            paddle.to_tensor([4.0], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c4d317d2c0091f0cdbcf7f9788698b06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_c4d317d2c0091f0cdbcf7f9788698b06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_bb89190f7a487956a4185509b9ccc802(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(176725.484375, dtype='float32').reshape([]),
            paddle.to_tensor([0.42551374435424805], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a06e3e5726019a9383974c445ed82d41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(116908.9140625, dtype='float32').reshape([]),
            paddle.to_tensor([0.42551374435424805], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b1b2fe400a1294adc1ce71812a6ac7f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67d7f8cfed71c4d3601fdbefae2a90db
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 152, 272], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_65b0c45d8679112e282521cae5d40f45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dee2530b07fc3e3f26a287a831e1e28d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11a6c2ce7213d45a7b8f7fce3e64d8a6
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3039810359477997], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a6ac806808fcbf9ac2b46c1f7fbfddc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_a6ac806808fcbf9ac2b46c1f7fbfddc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_82746842aee34a648d162a4993bc5dfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(52401.5390625, dtype='float32').reshape([]),
            paddle.to_tensor([0.27586016058921814], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9c37bc032b078c9961fdd492e848830a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(268655.5, dtype='float32').reshape([]),
            paddle.to_tensor([0.27586016058921814], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_81a9432f12e1c1e6a3eb2311ff170c13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_603d97710ed6a19cc34142e06ccee4c9
    def get_inputs(self):
        return [
            paddle.to_tensor([306.7394104003906], dtype='float32').reshape([1]),
            paddle.to_tensor(2434.0, dtype='float32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f932bc1bb949dc605d221289fbf3dc25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_f932bc1bb949dc605d221289fbf3dc25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_27894364c5eef25131335a2ff1243d34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(33165.703125, dtype='float32').reshape([]),
            paddle.to_tensor([0.3705717623233795], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eae73041ced26629de998ba339483e37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(15098.0625, dtype='float32').reshape([]),
            paddle.to_tensor([0.3705717623233795], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_545d921ce54972722f299efa19089ae2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b8e08c35b938a307e103ec0e962f34c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
            paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a591aaef7f5333704424b228d7fd7318(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.1558956503868103, 0.27804291248321533, 0.38011226058006287, 0.020000966265797615], [0.3080019950866699, 0.016835307702422142, 0.18333855271339417, 0.12878437340259552]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([[0.0657414123415947, 0.2821248769760132, 0.41800209879875183, 0.2077385038137436], [0.06425489485263824, 0.08323095738887787, 0.13659104704856873, 0.3990457355976105]], dtype='float32').reshape([2, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4850c1e2e60ea7a945cff314249a9a4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6d1d74352a5079d3dcd10e253979bfd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
            paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([6069, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8f0dfb8de58d3a290a2101ce3e1e2a1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
            paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a7e08227a125f3692f04448985f4022e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4007115662097931, 0.12639737129211426, 0.4112824499607086, 0.008094380609691143], [0.29197075963020325, 0.24120639264583588, 0.499051958322525, 0.09907882660627365]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([[0.27978193759918213, 0.12155638635158539, 0.46089935302734375, 0.44037988781929016], [0.11692929267883301, 0.10377798974514008, 0.16839845478534698, 0.04607399180531502]], dtype='float32').reshape([2, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_31d991ddc31e6d08482bcc84c03b3344(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[-0.01218876801431179], [-0.007885221391916275], [-0.07408457249403], [-0.14463911950588226], [-0.017956875264644623]], dtype='float32').reshape([5, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b5f3283f3f267a3a7c358cff3b14693c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.03584502637386322], [0.041892047971487045], [0.04215621203184128], [0.11972222477197647], [-0.01645669713616371]], dtype='float32').reshape([5, 1]),
            paddle.to_tensor([[0.02365626022219658], [0.03400682657957077], [-0.03192836046218872], [-0.02491689659655094], [-0.034413572400808334]], dtype='float32').reshape([5, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_535c0567e2f3bbbb9c42694343d5692e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11a6c2ce7213d45a7b8f7fce3e64d8a6
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3863242268562317], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_df13a428be372fa1e867c6ad008050a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e1651d47886b31cc4d74a01c4a69a08e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d558c26136203421c57b4b72437d23ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_126e1c1abeb0e1769c9c03280842532f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_126e1c1abeb0e1769c9c03280842532f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_7a63521ad0c59c53e2e94d5755d30e5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(-260738.65625, dtype='float32').reshape([]),
            paddle.to_tensor([0.3184764087200165], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_14a1aa5deb1d431218b84f1149f3dd62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(133816.90625, dtype='float32').reshape([]),
            paddle.to_tensor([0.3184764087200165], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_92db35c1a54496bbf761972a1804b6ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_92db35c1a54496bbf761972a1804b6ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_d025a29fac2c0205b7c07d9b388e9828(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(75233.65625, dtype='float32').reshape([]),
            paddle.to_tensor([0.3707900643348694], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2998351683913c56cf106a679de471b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(170275.46875, dtype='float32').reshape([]),
            paddle.to_tensor([0.3707900643348694], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_671fd901d7a48d579f3d9018711f48e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_671fd901d7a48d579f3d9018711f48e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_cdac2e0eb19d351224fa508b7173bb28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(-657314.5, dtype='float32').reshape([]),
            paddle.to_tensor([0.4422951340675354], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_643e8d6cd5d76dc7a16d2f080e280855(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(219193.84375, dtype='float32').reshape([]),
            paddle.to_tensor([0.4422951340675354], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_48495e8ced2315f2549f1757a39aef9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11a6c2ce7213d45a7b8f7fce3e64d8a6
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2676987051963806], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_65b0c45d8679112e282521cae5d40f45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8ebec33baf768fee56f0994f294de81d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(15.995766639709473, dtype='float32').reshape([]),
            paddle.to_tensor([3.0], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3999dd338e35ea7e376af8a1b9b43666(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
            paddle.uniform([20267, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([20267, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2a315fdbd49fc237ba883243237d21fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[-0.054475411772727966], [0.008385568857192993], [0.03203628212213516], [-0.04012038931250572]], dtype='float32').reshape([4, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2aa6c5b73ad00089c1965f5e7df71d9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.12263159453868866], [0.014178849756717682], [-0.023235611617565155], [0.08588115870952606]], dtype='float32').reshape([4, 1]),
            paddle.to_tensor([[0.0681561827659607], [0.022564418613910675], [0.008800670504570007], [0.04576076939702034]], dtype='float32').reshape([4, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7d89ade18fd38022e228b2101de3f481(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(3.7783987522125244, dtype='float32').reshape([]),
            paddle.to_tensor([7.0], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8ae5d16ad12f08a0b434294b25822fc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_8ae5d16ad12f08a0b434294b25822fc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_ad693fe09381b3c8f6c2447a39984006(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(17242.330078125, dtype='float32').reshape([]),
            paddle.to_tensor([0.4598577618598938], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3eaec148fec9050007f5a0303472795e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(29657.19921875, dtype='float32').reshape([]),
            paddle.to_tensor([0.4598577618598938], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a2e85f21b269387aaccaaf05daa5146f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11a6c2ce7213d45a7b8f7fce3e64d8a6
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.27220335602760315], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d5459f296c9b1a161fbad0628977fdf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(34.268775939941406, dtype='float32').reshape([]),
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0397f679ef047f77df7cf4861ef4eff1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
            paddle.uniform([6804, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([6804, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d558c26136203421c57b4b72437d23ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0b3678cf7736712aa05c8023c6b53cd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(234.89334106445312, dtype='float32').reshape([]),
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f15ff585951d7860623a9f32f80d8ee5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(137.79554748535156, dtype='float32').reshape([]),
            paddle.to_tensor([7.0], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_128f67565db93e81675dd337049efee6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_73ad1292c5bdfe6e4908735e0fa958d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_53e727419a5aef17c3e361360c563fcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_53e727419a5aef17c3e361360c563fcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e0e1e41ebece0ab9d4c052a641f0081
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_785269ed177d3d517650f00a54b11016(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(594974.5625, dtype='float32').reshape([]),
            paddle.to_tensor([0.39719903469085693], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_02d1080c8e5e9bcb4b615436b788ad05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c98db0a4df011e0d88394462a48e3c
    def get_inputs(self):
        return [
            paddle.to_tensor(236797.140625, dtype='float32').reshape([]),
            paddle.to_tensor([0.39719903469085693], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_73ad1292c5bdfe6e4908735e0fa958d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_891a275dac9f8869a5ead59ebd0cc998
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
                raise RuntimeError(f"file {__file__} panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()



if __name__ == '__main__':
    unittest.main()