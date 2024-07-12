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
class PrimitiveOp_c6c70f4c117e6d656c3a717f6e6ffa66(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_38abef7f5e2122f3bad5c38c167e6b2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6c70f4c117e6d656c3a717f6e6ffa66
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], dtype='int64').reshape([16, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d7483721ae0271580a1b7300bcf5ae29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6c70f4c117e6d656c3a717f6e6ffa66
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], dtype='int64').reshape([16, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_8414d764fca25664baa042b0402be286(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_acc0d4c6fa326732945f01a0d392aea6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8414d764fca25664baa042b0402be286
    def get_inputs(self):
        return [
            paddle.uniform([1778, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1778, 4, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_acc0d4c6fa326732945f01a0d392aea6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8414d764fca25664baa042b0402be286
    def get_inputs(self):
        return [
            paddle.uniform([1778, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1778, 4, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1f6eadf0058e9d4794cffec7eaf5073b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8414d764fca25664baa042b0402be286
    def get_inputs(self):
        return [
            paddle.uniform([5640, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[5640, 4, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1f6eadf0058e9d4794cffec7eaf5073b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8414d764fca25664baa042b0402be286
    def get_inputs(self):
        return [
            paddle.uniform([5640, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[5640, 4, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1c48da58b33ec49311959bcae90b3949(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6c70f4c117e6d656c3a717f6e6ffa66
    def get_inputs(self):
        return [
            paddle.uniform([36, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[36, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1c48da58b33ec49311959bcae90b3949(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6c70f4c117e6d656c3a717f6e6ffa66
    def get_inputs(self):
        return [
            paddle.uniform([36, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[36, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_3d2667e216b02829e75d11375fbc8cd9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_78b1abb92ab138b971c6faf700fc92bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d2667e216b02829e75d11375fbc8cd9
    def get_inputs(self):
        return [
            paddle.uniform([1797, 4, 19], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1797, 4, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_78b1abb92ab138b971c6faf700fc92bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d2667e216b02829e75d11375fbc8cd9
    def get_inputs(self):
        return [
            paddle.uniform([1797, 4, 19], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1797, 4, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_acaf2cbafa45dcd9b5801484b3a7888d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6c70f4c117e6d656c3a717f6e6ffa66
    def get_inputs(self):
        return [
            paddle.uniform([24, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], dtype='int64').reshape([24, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fc55a2c6acc8639b38d20b5ea3529f12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6c70f4c117e6d656c3a717f6e6ffa66
    def get_inputs(self):
        return [
            paddle.uniform([24, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], dtype='int64').reshape([24, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9e36df3dc93583a5ae8124666cfbbbf1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8414d764fca25664baa042b0402be286
    def get_inputs(self):
        return [
            paddle.uniform([1501, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1501, 4, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9e36df3dc93583a5ae8124666cfbbbf1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8414d764fca25664baa042b0402be286
    def get_inputs(self):
        return [
            paddle.uniform([1501, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1501, 4, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eb24ec14fe7d2274a32345e8eb09987f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6c70f4c117e6d656c3a717f6e6ffa66
    def get_inputs(self):
        return [
            paddle.uniform([4, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [0], [0], [0]], dtype='int64').reshape([4, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_753219e7e7ff7cbb3ca621928f4910ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6c70f4c117e6d656c3a717f6e6ffa66
    def get_inputs(self):
        return [
            paddle.uniform([4, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1], [1], [1], [1]], dtype='int64').reshape([4, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_143bd49454626e37f09757db15b93c54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8414d764fca25664baa042b0402be286
    def get_inputs(self):
        return [
            paddle.uniform([2078, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[2078, 4, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_143bd49454626e37f09757db15b93c54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8414d764fca25664baa042b0402be286
    def get_inputs(self):
        return [
            paddle.uniform([2078, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[2078, 4, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5db5698b373bc9c860d3ae2735574a2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8414d764fca25664baa042b0402be286
    def get_inputs(self):
        return [
            paddle.uniform([4653, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[4653, 4, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5db5698b373bc9c860d3ae2735574a2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8414d764fca25664baa042b0402be286
    def get_inputs(self):
        return [
            paddle.uniform([4653, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[4653, 4, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_573252b3661cebc029925a1b30451bf2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fe8788ada9c280ef2f91ef9427228c80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_573252b3661cebc029925a1b30451bf2
    def get_inputs(self):
        return [
            paddle.uniform([1, 2434, 81], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1, 2434, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2c90c48c031137259d6cec3e283aa608(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8414d764fca25664baa042b0402be286
    def get_inputs(self):
        return [
            paddle.uniform([1022, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1022, 4, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2c90c48c031137259d6cec3e283aa608(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8414d764fca25664baa042b0402be286
    def get_inputs(self):
        return [
            paddle.uniform([1022, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1022, 4, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4c80e3c86b7ac1f09fe7bb0f6c7fe2d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8414d764fca25664baa042b0402be286
    def get_inputs(self):
        return [
            paddle.uniform([2305, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[2305, 4, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4c80e3c86b7ac1f09fe7bb0f6c7fe2d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8414d764fca25664baa042b0402be286
    def get_inputs(self):
        return [
            paddle.uniform([2305, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[2305, 4, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4644f97e6111e2bbf6496ea1069f72a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8414d764fca25664baa042b0402be286
    def get_inputs(self):
        return [
            paddle.uniform([3053, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[3053, 4, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4644f97e6111e2bbf6496ea1069f72a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8414d764fca25664baa042b0402be286
    def get_inputs(self):
        return [
            paddle.uniform([3053, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[3053, 4, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_efa7aa2c1a0672466eee1aaacea129da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8414d764fca25664baa042b0402be286
    def get_inputs(self):
        return [
            paddle.uniform([3712, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[3712, 4, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_efa7aa2c1a0672466eee1aaacea129da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8414d764fca25664baa042b0402be286
    def get_inputs(self):
        return [
            paddle.uniform([3712, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[3712, 4, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fa2d57df00a8ce8a75f17a0f5ca2c613(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6c70f4c117e6d656c3a717f6e6ffa66
    def get_inputs(self):
        return [
            paddle.uniform([20, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], dtype='int64').reshape([20, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_80e345511411e95b829623eddc057a93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6c70f4c117e6d656c3a717f6e6ffa66
    def get_inputs(self):
        return [
            paddle.uniform([20, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], dtype='int64').reshape([20, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9c4c8a3b1c4b28aaeb0f6fbee66df620(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_573252b3661cebc029925a1b30451bf2
    def get_inputs(self):
        return [
            paddle.uniform([1, 8732, 21], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1, 8732, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ad66922bfbf1cbd8e16d1f0287495cbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8414d764fca25664baa042b0402be286
    def get_inputs(self):
        return [
            paddle.uniform([2102, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[2102, 4, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ad66922bfbf1cbd8e16d1f0287495cbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8414d764fca25664baa042b0402be286
    def get_inputs(self):
        return [
            paddle.uniform([2102, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[2102, 4, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cf7643f8203d5a638668966ee72807e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8414d764fca25664baa042b0402be286
    def get_inputs(self):
        return [
            paddle.uniform([4228, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[4228, 4, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cf7643f8203d5a638668966ee72807e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8414d764fca25664baa042b0402be286
    def get_inputs(self):
        return [
            paddle.uniform([4228, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[4228, 4, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_ccc075fb0220f08efee59f4d046e63e5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_be190d8985055c66deeafbcb6d2397d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccc075fb0220f08efee59f4d046e63e5
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], dtype='int64').reshape([16, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_67674a6bd752d3064e10dabedffd568c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccc075fb0220f08efee59f4d046e63e5
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], dtype='int64').reshape([16, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_0b94458e69a6463f4c539a67e23a61ca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e154e6492c459c7f67147b026ca0f1a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b94458e69a6463f4c539a67e23a61ca
    def get_inputs(self):
        return [
            paddle.uniform([1778, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1778, 4, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e154e6492c459c7f67147b026ca0f1a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b94458e69a6463f4c539a67e23a61ca
    def get_inputs(self):
        return [
            paddle.uniform([1778, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1778, 4, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_10239f056aa635eb20ad226ea128c1c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b94458e69a6463f4c539a67e23a61ca
    def get_inputs(self):
        return [
            paddle.uniform([5640, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[5640, 4, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_10239f056aa635eb20ad226ea128c1c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b94458e69a6463f4c539a67e23a61ca
    def get_inputs(self):
        return [
            paddle.uniform([5640, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[5640, 4, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_69ad4fbcf1fc4c48ab3ea685adbf71ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccc075fb0220f08efee59f4d046e63e5
    def get_inputs(self):
        return [
            paddle.uniform([36, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[36, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_69ad4fbcf1fc4c48ab3ea685adbf71ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccc075fb0220f08efee59f4d046e63e5
    def get_inputs(self):
        return [
            paddle.uniform([36, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[36, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_87334227e605964c9950e50cae941d55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b94458e69a6463f4c539a67e23a61ca
    def get_inputs(self):
        return [
            paddle.uniform([1797, 4, 19], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1797, 4, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_87334227e605964c9950e50cae941d55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b94458e69a6463f4c539a67e23a61ca
    def get_inputs(self):
        return [
            paddle.uniform([1797, 4, 19], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1797, 4, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ba721ffdf1544b74aea67813e5bc1167(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccc075fb0220f08efee59f4d046e63e5
    def get_inputs(self):
        return [
            paddle.uniform([24, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], dtype='int64').reshape([24, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e95762b3196bbc4ed3f943b65d262578(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccc075fb0220f08efee59f4d046e63e5
    def get_inputs(self):
        return [
            paddle.uniform([24, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], dtype='int64').reshape([24, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_75b2d69274266698b0a3f7cc221b55f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b94458e69a6463f4c539a67e23a61ca
    def get_inputs(self):
        return [
            paddle.uniform([1501, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1501, 4, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_75b2d69274266698b0a3f7cc221b55f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b94458e69a6463f4c539a67e23a61ca
    def get_inputs(self):
        return [
            paddle.uniform([1501, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1501, 4, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cefdb0c238b3fe16bc7e68e548f080ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccc075fb0220f08efee59f4d046e63e5
    def get_inputs(self):
        return [
            paddle.uniform([4, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [0], [0], [0]], dtype='int64').reshape([4, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f90158d491c86df0b481fea5eb14b969(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccc075fb0220f08efee59f4d046e63e5
    def get_inputs(self):
        return [
            paddle.uniform([4, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1], [1], [1], [1]], dtype='int64').reshape([4, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_75f2a6af0d700176ed7160ec8aacef46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b94458e69a6463f4c539a67e23a61ca
    def get_inputs(self):
        return [
            paddle.uniform([2078, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[2078, 4, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_75f2a6af0d700176ed7160ec8aacef46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b94458e69a6463f4c539a67e23a61ca
    def get_inputs(self):
        return [
            paddle.uniform([2078, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[2078, 4, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_132162ecf793805c96af678c31bc1aca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b94458e69a6463f4c539a67e23a61ca
    def get_inputs(self):
        return [
            paddle.uniform([4653, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[4653, 4, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_132162ecf793805c96af678c31bc1aca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b94458e69a6463f4c539a67e23a61ca
    def get_inputs(self):
        return [
            paddle.uniform([4653, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[4653, 4, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_44eddbc7e8e2b87b5bcb46ec5b889f2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b94458e69a6463f4c539a67e23a61ca
    def get_inputs(self):
        return [
            paddle.uniform([1, 2434, 81], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1, 2434, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e391980cdbb86a548a6fa9f65b7049ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b94458e69a6463f4c539a67e23a61ca
    def get_inputs(self):
        return [
            paddle.uniform([1022, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1022, 4, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e391980cdbb86a548a6fa9f65b7049ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b94458e69a6463f4c539a67e23a61ca
    def get_inputs(self):
        return [
            paddle.uniform([1022, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1022, 4, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_333df22742cf97199a40dfbe6f827f77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b94458e69a6463f4c539a67e23a61ca
    def get_inputs(self):
        return [
            paddle.uniform([2305, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[2305, 4, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_333df22742cf97199a40dfbe6f827f77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b94458e69a6463f4c539a67e23a61ca
    def get_inputs(self):
        return [
            paddle.uniform([2305, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[2305, 4, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c00f7d9c4080ba9795c13459951baad7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b94458e69a6463f4c539a67e23a61ca
    def get_inputs(self):
        return [
            paddle.uniform([3053, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[3053, 4, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c00f7d9c4080ba9795c13459951baad7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b94458e69a6463f4c539a67e23a61ca
    def get_inputs(self):
        return [
            paddle.uniform([3053, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[3053, 4, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b7669152be3a5ac8761876c084826487(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b94458e69a6463f4c539a67e23a61ca
    def get_inputs(self):
        return [
            paddle.uniform([3712, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[3712, 4, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b7669152be3a5ac8761876c084826487(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b94458e69a6463f4c539a67e23a61ca
    def get_inputs(self):
        return [
            paddle.uniform([3712, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[3712, 4, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_19f0b9883e8176f66ce03a7397947f22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccc075fb0220f08efee59f4d046e63e5
    def get_inputs(self):
        return [
            paddle.uniform([20, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], dtype='int64').reshape([20, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2c7de77865fe92b441bdef707c222569(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccc075fb0220f08efee59f4d046e63e5
    def get_inputs(self):
        return [
            paddle.uniform([20, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], dtype='int64').reshape([20, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e728507d5d2e1977977dbc96b8470d28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b94458e69a6463f4c539a67e23a61ca
    def get_inputs(self):
        return [
            paddle.uniform([1, 8732, 21], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[1, 8732, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9ab3c8c4e6cd885fcd4c71c226c5870b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b94458e69a6463f4c539a67e23a61ca
    def get_inputs(self):
        return [
            paddle.uniform([2102, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[2102, 4, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9ab3c8c4e6cd885fcd4c71c226c5870b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b94458e69a6463f4c539a67e23a61ca
    def get_inputs(self):
        return [
            paddle.uniform([2102, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[2102, 4, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8cdef7336d55b47ded3ad2e2353ca560(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b94458e69a6463f4c539a67e23a61ca
    def get_inputs(self):
        return [
            paddle.uniform([4228, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[4228, 4, 1], dtype='int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. stderr: \n{try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8cdef7336d55b47ded3ad2e2353ca560(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b94458e69a6463f4c539a67e23a61ca
    def get_inputs(self):
        return [
            paddle.uniform([4228, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.randint(low=0, high=3, shape=[4228, 4, 1], dtype='int64'),
        ]


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