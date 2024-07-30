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
class PrimitiveOp_97734b6b7d382115f363cc5c1905e99a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fa5815866ecb01299932929bbfe09159(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97734b6b7d382115f363cc5c1905e99a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ab0cc67d4dbd9b1e22eb881220b4ff88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97734b6b7d382115f363cc5c1905e99a
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_84b2fd6931c5f6221dfe6dd5b5a1db9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97734b6b7d382115f363cc5c1905e99a
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8587bb309ce294d182602814338d3f41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97734b6b7d382115f363cc5c1905e99a
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8d8df0291631e7cec9f6c34b8facd2af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97734b6b7d382115f363cc5c1905e99a
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d73a961d9f24f3c1fb0d190aa8f8e252(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97734b6b7d382115f363cc5c1905e99a
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_618cbd6ef04303a8c054832ee190c12c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97734b6b7d382115f363cc5c1905e99a
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_59e9419af1856e4fbd9eb3106e0f0782(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_acb8095bee4e77de6143c65b3135f7d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59e9419af1856e4fbd9eb3106e0f0782
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_755fffa2dc92e4bcb9d25ee5d551e50b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59e9419af1856e4fbd9eb3106e0f0782
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_4bb0b2bdf8d09fb43cb635e334464442(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.2'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d2c50a9ef4b6ffdac5497ab5824865d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb0b2bdf8d09fb43cb635e334464442
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_32e6d521c566dcd581bcc836dd4f0cfd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb0b2bdf8d09fb43cb635e334464442
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_547d2391ee94de10db45b67bb84e1993(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb0b2bdf8d09fb43cb635e334464442
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_2dffee48e482159da5592c170e18e0fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb0b2bdf8d09fb43cb635e334464442
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_3612bcf2222623603ffde0fae0320c61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb0b2bdf8d09fb43cb635e334464442
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9d8c9b37a3e7d847134a9bda3fa1654e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97734b6b7d382115f363cc5c1905e99a
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_f0396970c7397827c4aab685068869a2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.2'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_91b073870f36ad317f12a33f8a295ae3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0396970c7397827c4aab685068869a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_e92ccc91cd64ff20c8561f9890d7b4ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0396970c7397827c4aab685068869a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_5f8b368c1c0a1d058b53b21d0d77bd14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0396970c7397827c4aab685068869a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_cf10414d0fb43e72befcb66f819818c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0396970c7397827c4aab685068869a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_01739ea3058f35f42918630f222f62d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0396970c7397827c4aab685068869a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_e9dc317d50c394b831880c54d9149ec3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59e9419af1856e4fbd9eb3106e0f0782
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 1, 1], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_e8d41ed8b07f372922f4542031265508(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59e9419af1856e4fbd9eb3106e0f0782
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_c5f80d6693bd769c48c153e4235169f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59e9419af1856e4fbd9eb3106e0f0782
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_6543053f53f621c8d7a7cf8a7ab270cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59e9419af1856e4fbd9eb3106e0f0782
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_05a805de331b317fb9545f451062ef0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59e9419af1856e4fbd9eb3106e0f0782
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_6e92031398b8315715d0682939779a64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb0b2bdf8d09fb43cb635e334464442
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_eb4f0485ba716122382192188f84c862(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb0b2bdf8d09fb43cb635e334464442
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[84824712.0]], [[95284336.0]], [[97356176.0]], [[86963552.0]], [[90752768.0]], [[111518128.0]], [[60443328.0]], [[69558072.0]], [[83116816.0]], [[67342592.0]], [[62492792.0]], [[73973392.0]], [[81502136.0]], [[93819888.0]], [[90248640.0]], [[73473312.0]], [[54809660.0]], [[92469432.0]], [[75756552.0]], [[85136032.0]], [[87803664.0]], [[63362528.0]], [[69613936.0]], [[81063888.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
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
class TestPrimitiveOp_c820650df465132213a5c4015a67fed8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb0b2bdf8d09fb43cb635e334464442
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[81201720.0]], [[38125288.0]], [[63918780.0]], [[55985256.0]], [[70267152.0]], [[75871000.0]], [[42736488.0]], [[71267104.0]], [[42040300.0]], [[71283240.0]], [[68987504.0]], [[68790232.0]], [[71715136.0]], [[62418592.0]], [[26058970.0]], [[66058408.0]], [[50013980.0]], [[88508192.0]], [[60736324.0]], [[88217472.0]], [[104686536.0]], [[53525312.0]], [[31178990.0]], [[54727856.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
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
class TestPrimitiveOp_b1731101c1e3a06739ae002f47b96c5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb0b2bdf8d09fb43cb635e334464442
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[75172080.0]], [[50562620.0]], [[43074772.0]], [[80369144.0]], [[65907632.0]], [[68748352.0]], [[67553744.0]], [[54757080.0]], [[67288384.0]], [[62067808.0]], [[74261728.0]], [[76924936.0]], [[75200552.0]], [[42847376.0]], [[41021516.0]], [[87332256.0]], [[36179248.0]], [[65691928.0]], [[65849672.0]], [[87846360.0]], [[39229148.0]], [[76444512.0]], [[62106748.0]], [[45863828.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
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
class TestPrimitiveOp_6be3a14667b9511aac642ede53f19d53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb0b2bdf8d09fb43cb635e334464442
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[46086148.0]], [[58717484.0]], [[50499796.0]], [[63113952.0]], [[51760412.0]], [[67008480.0]], [[73700416.0]], [[84398440.0]], [[68939160.0]], [[58296428.0]], [[55653388.0]], [[66944688.0]], [[73501240.0]], [[81445776.0]], [[27382012.0]], [[61014936.0]], [[53223236.0]], [[60160192.0]], [[58104916.0]], [[87192784.0]], [[68687680.0]], [[89968136.0]], [[36128564.0]], [[61977412.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
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
class TestPrimitiveOp_63bebeaf966922c6f98eaa38faa3e68e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb0b2bdf8d09fb43cb635e334464442
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4236.40234375]], [[3443.68212890625]], [[2539.111572265625]], [[3912.30908203125]], [[4090.588623046875]], [[1974.7413330078125]], [[2570.740478515625]], [[3928.867431640625]], [[3607.357177734375]], [[3218.930908203125]], [[4531.0634765625]], [[3557.418212890625]], [[2731.403564453125]], [[3605.56494140625]], [[4537.787109375]], [[4473.8583984375]], [[2452.344970703125]], [[1982.04296875]], [[3093.23974609375]], [[2998.45703125]], [[2653.72216796875]], [[3086.792236328125]], [[3731.70556640625]], [[4389.189453125]]]], dtype='float32').reshape([1, 24, 1, 1]),
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
class TestPrimitiveOp_6232221bb0643df2f6d0d5247b2d4b68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb0b2bdf8d09fb43cb635e334464442
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_861c84f3d22129b667c41f4096a9937e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb0b2bdf8d09fb43cb635e334464442
    def get_inputs(self):
        return [
            paddle.uniform([1, 232, 1, 1], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_69bc863287c44806ac01c7a4418a5d2d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b8f5e8f83e61934c42bf2afaee4959d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69bc863287c44806ac01c7a4418a5d2d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.5673828125, 0.72216796875, 1.2744140625, 1.1357421875, 0.53466796875, 1.12109375, 0.87109375, 0.458251953125, 1.4306640625, 1.1728515625, 1.08203125, 0.5791015625, 1.154296875, 0.845703125, 0.82666015625, 1.0576171875]], dtype='float16').reshape([1, 16]),
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
class TestPrimitiveOp_b6e0eaf6ee2c646c6e565391a0b6cc0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69bc863287c44806ac01c7a4418a5d2d
    def get_inputs(self):
        return [
            paddle.uniform([1, 128], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_aaf6cb7b998071a3edbdf8fb56b053a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69bc863287c44806ac01c7a4418a5d2d
    def get_inputs(self):
        return [
            paddle.to_tensor([[6.82421875, 4.0, 4.08984375, 2.984375, 4.0390625, 4.96484375, 3.88671875, 3.19140625, 4.94921875, 4.171875, 5.03125, 3.287109375, 5.65234375, 3.001953125, 4.07421875, 3.951171875, 6.19140625, 5.5703125, 5.41796875, 4.11328125, 3.99609375, 2.841796875, 6.3125, 3.349609375]], dtype='float16').reshape([1, 24]),
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
class TestPrimitiveOp_b86031fa26410c6a2dadd6d17c610ef8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69bc863287c44806ac01c7a4418a5d2d
    def get_inputs(self):
        return [
            paddle.uniform([1, 192], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_b9364d6ba953562ab390f963ddad09a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69bc863287c44806ac01c7a4418a5d2d
    def get_inputs(self):
        return [
            paddle.uniform([1, 32], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_a20cbde069914a44d9a6f3bfcf535c62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69bc863287c44806ac01c7a4418a5d2d
    def get_inputs(self):
        return [
            paddle.uniform([1, 256], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_01eec7b2441304bafc860baf9bc9f62e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69bc863287c44806ac01c7a4418a5d2d
    def get_inputs(self):
        return [
            paddle.uniform([1, 64], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_62cc24b2bd18180625845ad1f25b478b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69bc863287c44806ac01c7a4418a5d2d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_4775099a04df757183ecaea93cd1a064(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69bc863287c44806ac01c7a4418a5d2d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_5d69ff9ac86c5b554290f29cedb7f064(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69bc863287c44806ac01c7a4418a5d2d
    def get_inputs(self):
        return [
            paddle.uniform([1, 768], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_321684bac3c8a9d216837dd0309b481e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59e9419af1856e4fbd9eb3106e0f0782
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_37df8734ce5fd16d9e1c4706058de2d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0396970c7397827c4aab685068869a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_91e407278c1068b8a3b1b603b6686e68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0396970c7397827c4aab685068869a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 1, 1], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_3be33116a8b23b986cf320774e0fae2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59e9419af1856e4fbd9eb3106e0f0782
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.9375]], [[5.08984375]], [[7.75]], [[6.4609375]], [[6.87890625]], [[9.2265625]], [[4.65625]], [[6.78515625]], [[4.59765625]], [[6.83984375]], [[6.08984375]], [[7.140625]], [[6.7109375]], [[7.03515625]], [[5.640625]], [[4.109375]]]], dtype='float16').reshape([1, 16, 1, 1]),
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
class TestPrimitiveOp_a7d448f7808fc47f820464ecf52149fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59e9419af1856e4fbd9eb3106e0f0782
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_5d4d8a077106554a4d5e2baeb79f51f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59e9419af1856e4fbd9eb3106e0f0782
    def get_inputs(self):
        return [
            paddle.uniform([1, 28, 1, 1], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_8a1d84b4fc5a313e9e92b62255159016(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59e9419af1856e4fbd9eb3106e0f0782
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 1, 1], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_f96be59b45d2c40b29074582685474df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59e9419af1856e4fbd9eb3106e0f0782
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_d22ddb0342c2c2d9bb454fd249be893a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb0b2bdf8d09fb43cb635e334464442
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_e23ea76b1c25b05e6944df7118b32e42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb0b2bdf8d09fb43cb635e334464442
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f4fd8f4d6433afdf445819d5cec8bb8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb0b2bdf8d09fb43cb635e334464442
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_aa02bb2d13d657e59d4457fc02f17d29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb0b2bdf8d09fb43cb635e334464442
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b51cb6cc233b4a25a58fdb0dfac06cb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0396970c7397827c4aab685068869a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_8ada399a9cbafb9e588c4560d20930a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0396970c7397827c4aab685068869a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_99cadf4051ea0c173963d4e540ca3ad9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0396970c7397827c4aab685068869a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_88602d9a810452bcb5c1b72b4db8e8b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0396970c7397827c4aab685068869a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 1, 1], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_c674b6619a077e863fd7517752d395f7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d4af904c83b19fc4f3b532e95e195664(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c674b6619a077e863fd7517752d395f7
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.8988653421401978, 1.3468341827392578, 1.0325161218643188, 0.8475977778434753, 1.0868228673934937, 1.0237349271774292, 0.6818506717681885, 1.054220199584961, 1.2488194704055786, 0.9565609693527222, 0.7879104018211365, 0.9833962917327881, 1.0298106670379639, 0.9270545244216919, 0.9992638230323792, 1.0418517589569092]], dtype='float32').reshape([1, 16]),
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
class TestPrimitiveOp_4e8461aeb28f7fbb6f31b8c919fa9333(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c674b6619a077e863fd7517752d395f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d32c60b1b5871ba1f640aa58d9694ad8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c674b6619a077e863fd7517752d395f7
    def get_inputs(self):
        return [
            paddle.to_tensor([[6.067137241363525, 7.7949604988098145, 8.568772315979004, 5.087800979614258, 6.020412921905518, 5.8264288902282715, 8.395882606506348, 7.712253093719482, 5.076632976531982, 9.574555397033691, 9.779694557189941, 5.533529758453369, 6.446454048156738, 11.528704643249512, 4.776479244232178, 7.113524913787842, 8.277176856994629, 8.501121520996094, 7.05758810043335, 6.9912800788879395, 6.848237037658691, 7.410258769989014, 5.192013740539551, 8.960549354553223]], dtype='float32').reshape([1, 24]),
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
class TestPrimitiveOp_112194478dfc48f4f416ab1cf32a7723(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c674b6619a077e863fd7517752d395f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 192], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a28a3076036491bc9220c59cbefc7071(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c674b6619a077e863fd7517752d395f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 32], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a16cb7d9ebac39a4a3d6428de29dfb9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c674b6619a077e863fd7517752d395f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b4ab12e5f3ce805eaddc3f8425d05031(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c674b6619a077e863fd7517752d395f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_27481eccc934b50a538dbd3b76139979(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c674b6619a077e863fd7517752d395f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 512], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_aa55f37f9bbfa47e317385a9af37b4c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c674b6619a077e863fd7517752d395f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_e96d9d510edb21333bc86980353c9729(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c674b6619a077e863fd7517752d395f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 768], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_97bd8456926a4edc28c5a05029d9494a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0396970c7397827c4aab685068869a2
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1332.0]], [[2170.0]], [[2004.0]], [[2820.0]], [[2112.0]], [[1353.0]], [[1429.0]], [[2288.0]], [[1353.0]], [[1964.0]], [[1783.0]], [[1790.0]], [[850.0]], [[2420.0]], [[1637.0]], [[2230.0]], [[1541.0]], [[1691.0]], [[1666.0]], [[937.0]], [[1607.0]], [[2182.0]], [[1960.0]], [[2072.0]]]], dtype='float16').reshape([1, 24, 1, 1]),
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
class TestPrimitiveOp_bdb404c887b05322be2a187a774191bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0396970c7397827c4aab685068869a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_104fd76d1e186e141c3ea4c76e58518b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0396970c7397827c4aab685068869a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 232, 1, 1], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_17233b18cfbdb47636119f5ca29ff80c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59e9419af1856e4fbd9eb3106e0f0782
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_947cb1fb53a2f4f0664d538c474a9a82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59e9419af1856e4fbd9eb3106e0f0782
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_40b17b77616bc049812977d38d0133a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb0b2bdf8d09fb43cb635e334464442
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[279891040.0]], [[264132832.0]], [[311595680.0]], [[277734592.0]], [[307796352.0]], [[228904480.0]], [[373297632.0]], [[301044096.0]], [[296394208.0]], [[449432768.0]], [[349804352.0]], [[395429568.0]], [[324786560.0]], [[288108608.0]], [[446277184.0]], [[270903296.0]], [[290407008.0]], [[251125408.0]], [[470202560.0]], [[280867840.0]], [[476148288.0]], [[330308224.0]], [[237694176.0]], [[344678272.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
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
class TestPrimitiveOp_65feb6a45b876822d2c0f52201101c9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb0b2bdf8d09fb43cb635e334464442
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[130853312.0]], [[306298176.0]], [[286608736.0]], [[282970560.0]], [[268030672.0]], [[283060544.0]], [[362496256.0]], [[209395792.0]], [[289811456.0]], [[277383808.0]], [[309473728.0]], [[268407680.0]], [[154358272.0]], [[255280448.0]], [[252437632.0]], [[175629968.0]], [[356006848.0]], [[322790464.0]], [[308607072.0]], [[201436176.0]], [[221079680.0]], [[246881840.0]], [[297502656.0]], [[304853632.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
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
class TestPrimitiveOp_1ccbe4b3165c538af7a8e4b72a79efa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb0b2bdf8d09fb43cb635e334464442
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[318694048.0]], [[191371328.0]], [[343039808.0]], [[434812480.0]], [[336776896.0]], [[387936800.0]], [[279104224.0]], [[365404928.0]], [[369353536.0]], [[341404544.0]], [[283913760.0]], [[131918192.0]], [[263453248.0]], [[325897440.0]], [[374120704.0]], [[234163024.0]], [[318166816.0]], [[224596576.0]], [[231163776.0]], [[286103616.0]], [[313182784.0]], [[241216240.0]], [[303616640.0]], [[299664256.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
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
class TestPrimitiveOp_d5b034c9b4b3a92afd5de26afbbcb593(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb0b2bdf8d09fb43cb635e334464442
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[202415200.0]], [[414818880.0]], [[147580832.0]], [[217471200.0]], [[177146480.0]], [[258451536.0]], [[378640000.0]], [[384011200.0]], [[220621344.0]], [[311907264.0]], [[358289920.0]], [[332792832.0]], [[221420640.0]], [[268747872.0]], [[271747712.0]], [[176029312.0]], [[257953280.0]], [[268968448.0]], [[260324224.0]], [[220420208.0]], [[278406336.0]], [[326242944.0]], [[298994656.0]], [[351935360.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
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
class TestPrimitiveOp_ba2771e1bf4c75d39b8efcc6604df2e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97734b6b7d382115f363cc5c1905e99a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.759235858917236]], [[14.225831031799316]], [[19.027368545532227]], [[12.69971752166748]], [[13.102041244506836]], [[14.917290687561035]], [[6.757692337036133]], [[15.396809577941895]], [[16.261119842529297]], [[15.060494422912598]], [[14.572915077209473]], [[11.219926834106445]], [[12.17980670928955]], [[13.219276428222656]], [[14.74959945678711]], [[6.9188432693481445]]]], dtype='float32').reshape([1, 16, 1, 1]),
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
class TestPrimitiveOp_cede7a005e758c72caba03ad4c6b0d5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97734b6b7d382115f363cc5c1905e99a
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8998bd8b78c15e74f85c8823de43f827(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97734b6b7d382115f363cc5c1905e99a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[386696.71875]], [[238397.765625]], [[215312.25]], [[270892.8125]], [[313143.3125]], [[368320.28125]], [[282008.46875]], [[301115.34375]], [[442644.1875]], [[355851.21875]], [[369400.9375]], [[293284.0]], [[259967.015625]], [[299579.125]], [[339983.53125]], [[172553.265625]], [[202926.5]], [[392458.84375]], [[317232.59375]], [[270807.03125]], [[414629.65625]], [[271966.0]], [[191804.296875]], [[307798.25]], [[355791.0625]], [[285885.0]], [[296692.09375]], [[451372.625]]]], dtype='float32').reshape([1, 28, 1, 1]),
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
class TestPrimitiveOp_de2b5afb9190a3752bdb15ff53af91bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97734b6b7d382115f363cc5c1905e99a
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_635ed2b12f87bd13ab8022b5847e6bed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97734b6b7d382115f363cc5c1905e99a
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5ca774e4e86c233f4b5a6d6583096c47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97734b6b7d382115f363cc5c1905e99a
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_bd2d98309e339ae3ebba31733cab89a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97734b6b7d382115f363cc5c1905e99a
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_7854cd45961d6917ef21c9ac784f90e2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a91976c48c471df3f6ea1bb69ca74840(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7854cd45961d6917ef21c9ac784f90e2
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_195bd8dbc9d698876f32a55fd65c1b47(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9608ab13cec555b82cefadcd32b1b7b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_195bd8dbc9d698876f32a55fd65c1b47
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_4622bd74e78da64c75dd2530e0649ca4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 44, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0e10a5a078f70a02452dcfbf8d4293f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4622bd74e78da64c75dd2530e0649ca4
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 1, 1], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_9d41cd4fbb77e2653f30f8a888a9398a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3baf1739e55e595b8ea522851dc9df93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d41cd4fbb77e2653f30f8a888a9398a
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_fc6f00e74bc677447af0af34d56acffb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 120, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9e49e1ffd05885e3c32a04e33660beac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc6f00e74bc677447af0af34d56acffb
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_fa82877cf78deeaeaed502e16b2599fe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_df35e397be462f92c55b17ef75390bac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fa82877cf78deeaeaed502e16b2599fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_bf7bd1b77f7344a895e0d4d6eaa606b3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_292fe7725bc18294b85335ab968d2706(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf7bd1b77f7344a895e0d4d6eaa606b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_7b1d08fc9846c8d38993c00ebf25d319(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a05a26b5449acca72b43ccaf2020c3af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b1d08fc9846c8d38993c00ebf25d319
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_e3cff2e8ebea103f2b1cefbb2d0e38f2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_64ce5c98083d43d666b0cd341f388509(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3cff2e8ebea103f2b1cefbb2d0e38f2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_c618d1f71f04eba8334909435ca33a01(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.2'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 40, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fc44b2b11cfad0cff530aebc4a1df3d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c618d1f71f04eba8334909435ca33a01
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_8e8f71ff4530151fbd55547dac3a2cae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.2'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9e226fa8e01668c13461326bac0f2eb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e8f71ff4530151fbd55547dac3a2cae
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_91ed9c3713cadcfbfaf4e56b860acbb9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.2'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_798f0aef641ade1529a8c00285a90619(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91ed9c3713cadcfbfaf4e56b860acbb9
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_0a16901057441daee9a7a4b2b224fc15(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.2'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 336, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ec6932be50f3b29d5e4f7eba75ac1fd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a16901057441daee9a7a4b2b224fc15
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 1, 1], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_1713ffa0f68bbc205ded657c21fbde6f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.2'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 480, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_27641930da19ffcd48670c999171aec3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1713ffa0f68bbc205ded657c21fbde6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_ce0aa4bb218b3e566fc286676f0b5fe1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 384, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3e6d89cbd8e7adbef527a26576beef01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce0aa4bb218b3e566fc286676f0b5fe1
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_b08abd2d6b31e9df636593a15d43e96b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.2'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 72, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fb6c253224e85163d5c41f58fb1e6ee6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b08abd2d6b31e9df636593a15d43e96b
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_65f396cc8011288f993b54537676be75(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.2'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 120, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_11d5bc09f895d69fc9d67b863e3f7912(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65f396cc8011288f993b54537676be75
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_b33fbcb6c33c1f36eaa202e878965345(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.2'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 480, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_73e3a3260bc130e8d73901411d5c1931(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b33fbcb6c33c1f36eaa202e878965345
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_f7ca8e8ac255995eef9936fc7baeead4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.2'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 672, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b1407c3fdbe9212add4afacbeb46d271(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7ca8e8ac255995eef9936fc7baeead4
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_847a909d01627c73fd043d8d1f133739(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.2'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 960, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5d5d2bbc0b681fff0d09bcaa4c06072e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_847a909d01627c73fd043d8d1f133739
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_aa3e1c04707eabbd557eb13606b7bddc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 44, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_882a0852d211b4bfa11f41ce5bb33b99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa3e1c04707eabbd557eb13606b7bddc
    def get_inputs(self):
        return [
            paddle.uniform([1, 44, 1, 1], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_d14e8fb73e99c3fe9388fe661d588373(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c20ebf9a346128a8ff28ee9b254e84ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d14e8fb73e99c3fe9388fe661d588373
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_b8453f2e048bd82bfa9c232e6974c68b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 120, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3be9b070c365be06b71b84397f639ce3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8453f2e048bd82bfa9c232e6974c68b
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_ab5c59c3d3cd4bcf4f1dcfa858879a7d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f875a962be044439e59700697152cf61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab5c59c3d3cd4bcf4f1dcfa858879a7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_a8361730f7d0eeb86165a2164f8a1d6a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f68467b1442607ad42bfe1f951f538c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8361730f7d0eeb86165a2164f8a1d6a
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_1a405aac43d6823aded686f3010bceec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.2'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c715fa9e899d225a40f2b15fc6bb1a07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a405aac43d6823aded686f3010bceec
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_6fd30215be3f793c10a42ecb8429dd79(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.2'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 24, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d68ec973ab383c9813b3b392307fbf6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fd30215be3f793c10a42ecb8429dd79
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[84824712.0]], [[95284336.0]], [[97356176.0]], [[86963552.0]], [[90752768.0]], [[111518128.0]], [[60443328.0]], [[69558072.0]], [[83116816.0]], [[67342592.0]], [[62492792.0]], [[73973392.0]], [[81502136.0]], [[93819888.0]], [[90248640.0]], [[73473312.0]], [[54809660.0]], [[92469432.0]], [[75756552.0]], [[85136032.0]], [[87803664.0]], [[63362528.0]], [[69613936.0]], [[81063888.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
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
class TestPrimitiveOp_873aac264eeb231b73667802129a3358(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fd30215be3f793c10a42ecb8429dd79
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[81201720.0]], [[38125288.0]], [[63918780.0]], [[55985256.0]], [[70267152.0]], [[75871000.0]], [[42736488.0]], [[71267104.0]], [[42040300.0]], [[71283240.0]], [[68987504.0]], [[68790232.0]], [[71715136.0]], [[62418592.0]], [[26058970.0]], [[66058408.0]], [[50013980.0]], [[88508192.0]], [[60736324.0]], [[88217472.0]], [[104686536.0]], [[53525312.0]], [[31178990.0]], [[54727856.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
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
class TestPrimitiveOp_5e400947a7ead41f958a8fe9045fe526(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fd30215be3f793c10a42ecb8429dd79
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[75172080.0]], [[50562620.0]], [[43074772.0]], [[80369144.0]], [[65907632.0]], [[68748352.0]], [[67553744.0]], [[54757080.0]], [[67288384.0]], [[62067808.0]], [[74261728.0]], [[76924936.0]], [[75200552.0]], [[42847376.0]], [[41021516.0]], [[87332256.0]], [[36179248.0]], [[65691928.0]], [[65849672.0]], [[87846360.0]], [[39229148.0]], [[76444512.0]], [[62106748.0]], [[45863828.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
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
class TestPrimitiveOp_1fde2b64491d6253ff820ee366e5dd02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fd30215be3f793c10a42ecb8429dd79
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[46086148.0]], [[58717484.0]], [[50499796.0]], [[63113952.0]], [[51760412.0]], [[67008480.0]], [[73700416.0]], [[84398440.0]], [[68939160.0]], [[58296428.0]], [[55653388.0]], [[66944688.0]], [[73501240.0]], [[81445776.0]], [[27382012.0]], [[61014936.0]], [[53223236.0]], [[60160192.0]], [[58104916.0]], [[87192784.0]], [[68687680.0]], [[89968136.0]], [[36128564.0]], [[61977412.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
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
class TestPrimitiveOp_91f07a97dec0890601b974429e1fc779(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fd30215be3f793c10a42ecb8429dd79
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4236.40234375]], [[3443.68212890625]], [[2539.111572265625]], [[3912.30908203125]], [[4090.588623046875]], [[1974.7413330078125]], [[2570.740478515625]], [[3928.867431640625]], [[3607.357177734375]], [[3218.930908203125]], [[4531.0634765625]], [[3557.418212890625]], [[2731.403564453125]], [[3605.56494140625]], [[4537.787109375]], [[4473.8583984375]], [[2452.344970703125]], [[1982.04296875]], [[3093.23974609375]], [[2998.45703125]], [[2653.72216796875]], [[3086.792236328125]], [[3731.70556640625]], [[4389.189453125]]]], dtype='float32').reshape([1, 24, 1, 1]),
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

class PrimitiveOp_856c7af0a11011270775b5ebb34d9dac(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.2'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 168, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3f9e19c87c59fa0965dea2ff8f7c3cc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_856c7af0a11011270775b5ebb34d9dac
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_39f1743008024587cb8d0e70724b4141(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.2'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 232, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cc19da54a9d38bb56ecfb82f84563c20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39f1743008024587cb8d0e70724b4141
    def get_inputs(self):
        return [
            paddle.uniform([1, 232, 1, 1], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_efd47f5f6a28b485ab345c9db14fdfb9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cb637fe023e130c175cee76318dd24a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efd47f5f6a28b485ab345c9db14fdfb9
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.5673828125, 0.72216796875, 1.2744140625, 1.1357421875, 0.53466796875, 1.12109375, 0.87109375, 0.458251953125, 1.4306640625, 1.1728515625, 1.08203125, 0.5791015625, 1.154296875, 0.845703125, 0.82666015625, 1.0576171875]], dtype='float16').reshape([1, 16]),
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

class PrimitiveOp_014a3ba7359bb15f83f3a0f68fa06690(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b6e566d240c115973ddbaa4442677395(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_014a3ba7359bb15f83f3a0f68fa06690
    def get_inputs(self):
        return [
            paddle.uniform([1, 128], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_edf8b9d802a352bb38bc434f65b334a0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 24], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a0161c27dbb05873f79e8fd6a9927179(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_edf8b9d802a352bb38bc434f65b334a0
    def get_inputs(self):
        return [
            paddle.to_tensor([[6.82421875, 4.0, 4.08984375, 2.984375, 4.0390625, 4.96484375, 3.88671875, 3.19140625, 4.94921875, 4.171875, 5.03125, 3.287109375, 5.65234375, 3.001953125, 4.07421875, 3.951171875, 6.19140625, 5.5703125, 5.41796875, 4.11328125, 3.99609375, 2.841796875, 6.3125, 3.349609375]], dtype='float16').reshape([1, 24]),
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

class PrimitiveOp_cb0dadac809baf713a0ca255351ac2c3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1e915e50fd91384c52f582099556ca2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb0dadac809baf713a0ca255351ac2c3
    def get_inputs(self):
        return [
            paddle.uniform([1, 192], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_bbe0fecb4695ebbbdcce9631d186a63a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_236caa4c7be4dea8a01a8a1900244bce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbe0fecb4695ebbbdcce9631d186a63a
    def get_inputs(self):
        return [
            paddle.uniform([1, 32], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_9f7f06e782d117b9eb9226370a0f85d5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a356ff44675445f4dc4a523546baa556(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f7f06e782d117b9eb9226370a0f85d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 256], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_1462e4867016d11858ba3f9f575727eb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3fc15c42ed81338c59714a71a8d21cb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1462e4867016d11858ba3f9f575727eb
    def get_inputs(self):
        return [
            paddle.uniform([1, 64], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_12329987dd85d6e88541077d2e0531b4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_45bc313f9167ba736d033866f3e20161(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12329987dd85d6e88541077d2e0531b4
    def get_inputs(self):
        return [
            paddle.uniform([1, 512], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_06af3ef6ea14929a639d198e7a2c5f8d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1024], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_395ed4998a15190ec21ad496223ca20a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06af3ef6ea14929a639d198e7a2c5f8d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_6fd67e537748788337681c4fe2981600(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 768], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_db6f19b2a025c73dd204ca31ecb54d52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fd67e537748788337681c4fe2981600
    def get_inputs(self):
        return [
            paddle.uniform([1, 768], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_1d2100b6702291e9d914025d985adfb0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 384, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_48141f958a982a65ef57333f66813f45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d2100b6702291e9d914025d985adfb0
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_d06f3748999ec7d8fefd3b21840fc923(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.2'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7356caeb7e65bb699199ded95b90b81d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d06f3748999ec7d8fefd3b21840fc923
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_759275727931e58b5e8b8c3425245c81(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.2'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 24, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_35c1f4d40a8573d1e401563cab53538f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_759275727931e58b5e8b8c3425245c81
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 1, 1], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_43169cad73b69393a37f1d7f64d519c0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_85cdf32bacaf6a8ab3fd1bcbbe14c747(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43169cad73b69393a37f1d7f64d519c0
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.9375]], [[5.08984375]], [[7.75]], [[6.4609375]], [[6.87890625]], [[9.2265625]], [[4.65625]], [[6.78515625]], [[4.59765625]], [[6.83984375]], [[6.08984375]], [[7.140625]], [[6.7109375]], [[7.03515625]], [[5.640625]], [[4.109375]]]], dtype='float16').reshape([1, 16, 1, 1]),
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

class PrimitiveOp_fe11f0701e4adff2fc66668ddac4a75c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0775c4e4a8bf15223d7e7d093569bf3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe11f0701e4adff2fc66668ddac4a75c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_a01b7205b294238a4ef6b90533bd2b38(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 28, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_041981918ba70ddba1e0f8a71024fc16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a01b7205b294238a4ef6b90533bd2b38
    def get_inputs(self):
        return [
            paddle.uniform([1, 28, 1, 1], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_529c8e3528bc3446b0fec29399cccd37(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 56, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b4c69e2e271c7e9dd6681bd48d49aea1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_529c8e3528bc3446b0fec29399cccd37
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 1, 1], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_788918e3bf82a7eb721cbc696ac12426(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 60, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4cafb4af97e853eaea432db2d8984d83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_788918e3bf82a7eb721cbc696ac12426
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_df0f21df864fa30c2c119fdc7b873c1a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.2'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 72, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ca9fbb39159c5350dce019be77d1600d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df0f21df864fa30c2c119fdc7b873c1a
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_028844ce25cf2983eb70e21c7f7d1f7e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.2'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 120, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_027ca669e4a9b4facfdccaa875700928(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_028844ce25cf2983eb70e21c7f7d1f7e
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_c3780fee1f7996e57b9a3f9edc248eea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.2'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 672, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6d44d8093feeb45443f5436d1ad53645(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3780fee1f7996e57b9a3f9edc248eea
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_0744c075fb69a527ab46276c4d2d79e5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.2'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 960, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3a3980cba4e859fcce413890f94ff6ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0744c075fb69a527ab46276c4d2d79e5
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_5501d02b9cc35073fc846584980a2278(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.2'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 40, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e8632c0ec2c1e968386b641279b523ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5501d02b9cc35073fc846584980a2278
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_85dfdc0bf34f888692cc7904de8a8f0d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.2'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fe598ceebea152fbffb7e787ebf45c2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85dfdc0bf34f888692cc7904de8a8f0d
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_51d8f4cb102fe30c6818f788caa2d080(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.2'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_968c6dd93e8b023edbb8ed9a2c91a7e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51d8f4cb102fe30c6818f788caa2d080
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_ea4e108f279c16818dd8f5f7512f9177(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.2'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 336, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_50ea1d5a1811ddc5acdfe1361ae05d58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea4e108f279c16818dd8f5f7512f9177
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 1, 1], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_74a32c890a9f210607759279281d21e1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_70b7c4c696e12be38acb7cb518741c2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74a32c890a9f210607759279281d21e1
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.8988653421401978, 1.3468341827392578, 1.0325161218643188, 0.8475977778434753, 1.0868228673934937, 1.0237349271774292, 0.6818506717681885, 1.054220199584961, 1.2488194704055786, 0.9565609693527222, 0.7879104018211365, 0.9833962917327881, 1.0298106670379639, 0.9270545244216919, 0.9992638230323792, 1.0418517589569092]], dtype='float32').reshape([1, 16]),
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

class PrimitiveOp_39b678d44be9042e14b22ce5931da428(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_438612695c1b1aa33b58a96b217be784(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39b678d44be9042e14b22ce5931da428
    def get_inputs(self):
        return [
            paddle.uniform([1, 128], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_fed9994e91a2d9cc8557babdb2849cdd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4fae918169f768fbcb286be9a15544e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fed9994e91a2d9cc8557babdb2849cdd
    def get_inputs(self):
        return [
            paddle.to_tensor([[6.067137241363525, 7.7949604988098145, 8.568772315979004, 5.087800979614258, 6.020412921905518, 5.8264288902282715, 8.395882606506348, 7.712253093719482, 5.076632976531982, 9.574555397033691, 9.779694557189941, 5.533529758453369, 6.446454048156738, 11.528704643249512, 4.776479244232178, 7.113524913787842, 8.277176856994629, 8.501121520996094, 7.05758810043335, 6.9912800788879395, 6.848237037658691, 7.410258769989014, 5.192013740539551, 8.960549354553223]], dtype='float32').reshape([1, 24]),
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

class PrimitiveOp_b8cec2b0dc318268b3d49e814c8070df(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cb2e00edc935ba74eb51c8b0646028bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8cec2b0dc318268b3d49e814c8070df
    def get_inputs(self):
        return [
            paddle.uniform([1, 192], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_ace7ee5e6411ba534401f08f3528c213(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e9c94068619a5d28985511afd8735234(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ace7ee5e6411ba534401f08f3528c213
    def get_inputs(self):
        return [
            paddle.uniform([1, 32], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_a4693941ce6abeaa0179a557ba0ec1c3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fb0f6f3b249e460f3f7629bc1ef054e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4693941ce6abeaa0179a557ba0ec1c3
    def get_inputs(self):
        return [
            paddle.uniform([1, 256], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_0d059a4cbbdd56976a49318955874975(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8d0e73fb4a3dd9b112a04b496b40db66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d059a4cbbdd56976a49318955874975
    def get_inputs(self):
        return [
            paddle.uniform([1, 64], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_81e51a3bb48a003f069732cdb2f65c51(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a90bb39f69d7d43fb5cd544bc70ee5fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81e51a3bb48a003f069732cdb2f65c51
    def get_inputs(self):
        return [
            paddle.uniform([1, 512], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_f353c02e3ec560d2629c0bf54621b8da(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_729722213ed9f9b3ddc3ff528e20af30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f353c02e3ec560d2629c0bf54621b8da
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_06c4d75d90303e62dbfa10fc6ec30ab6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fe571d71001ae9dbf9cb66b72bb9db3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06c4d75d90303e62dbfa10fc6ec30ab6
    def get_inputs(self):
        return [
            paddle.uniform([1, 768], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_3b0f1bd9a30b4d1bc75cd72624e99a85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_759275727931e58b5e8b8c3425245c81
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1332.0]], [[2170.0]], [[2004.0]], [[2820.0]], [[2112.0]], [[1353.0]], [[1429.0]], [[2288.0]], [[1353.0]], [[1964.0]], [[1783.0]], [[1790.0]], [[850.0]], [[2420.0]], [[1637.0]], [[2230.0]], [[1541.0]], [[1691.0]], [[1666.0]], [[937.0]], [[1607.0]], [[2182.0]], [[1960.0]], [[2072.0]]]], dtype='float16').reshape([1, 24, 1, 1]),
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

class PrimitiveOp_48824195393c171314889060b028b209(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.2'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 168, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3dbe5afee69c2ae7425e3a2af88c7df2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48824195393c171314889060b028b209
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_f1b949c950e208eea7a747f49a2b4562(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.2'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 232, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6511f5a8fe6a9bb415d2ecd4d1aca426(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f1b949c950e208eea7a747f49a2b4562
    def get_inputs(self):
        return [
            paddle.uniform([1, 232, 1, 1], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_78615fd4e1170aaf24b69d1c2419b9d6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_277b9dd4a8719ce02ada06af8b9b74dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78615fd4e1170aaf24b69d1c2419b9d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_748d845577940106ae14b83b41b7b8a4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5d3f631ae14bac843550da142a573105(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_748d845577940106ae14b83b41b7b8a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_e28cd2d1d6dbd0e9e041ecd35c97b49d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fd30215be3f793c10a42ecb8429dd79
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[279891040.0]], [[264132832.0]], [[311595680.0]], [[277734592.0]], [[307796352.0]], [[228904480.0]], [[373297632.0]], [[301044096.0]], [[296394208.0]], [[449432768.0]], [[349804352.0]], [[395429568.0]], [[324786560.0]], [[288108608.0]], [[446277184.0]], [[270903296.0]], [[290407008.0]], [[251125408.0]], [[470202560.0]], [[280867840.0]], [[476148288.0]], [[330308224.0]], [[237694176.0]], [[344678272.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
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
class TestPrimitiveOp_0e20b36eb6d2bee98175e872114e0c98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fd30215be3f793c10a42ecb8429dd79
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[130853312.0]], [[306298176.0]], [[286608736.0]], [[282970560.0]], [[268030672.0]], [[283060544.0]], [[362496256.0]], [[209395792.0]], [[289811456.0]], [[277383808.0]], [[309473728.0]], [[268407680.0]], [[154358272.0]], [[255280448.0]], [[252437632.0]], [[175629968.0]], [[356006848.0]], [[322790464.0]], [[308607072.0]], [[201436176.0]], [[221079680.0]], [[246881840.0]], [[297502656.0]], [[304853632.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
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
class TestPrimitiveOp_53a1c926bcce2a6834a901bfd8e5dee8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fd30215be3f793c10a42ecb8429dd79
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[318694048.0]], [[191371328.0]], [[343039808.0]], [[434812480.0]], [[336776896.0]], [[387936800.0]], [[279104224.0]], [[365404928.0]], [[369353536.0]], [[341404544.0]], [[283913760.0]], [[131918192.0]], [[263453248.0]], [[325897440.0]], [[374120704.0]], [[234163024.0]], [[318166816.0]], [[224596576.0]], [[231163776.0]], [[286103616.0]], [[313182784.0]], [[241216240.0]], [[303616640.0]], [[299664256.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
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
class TestPrimitiveOp_d4c3b1a575c831f305f83f6bfa94e9b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6fd30215be3f793c10a42ecb8429dd79
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[202415200.0]], [[414818880.0]], [[147580832.0]], [[217471200.0]], [[177146480.0]], [[258451536.0]], [[378640000.0]], [[384011200.0]], [[220621344.0]], [[311907264.0]], [[358289920.0]], [[332792832.0]], [[221420640.0]], [[268747872.0]], [[271747712.0]], [[176029312.0]], [[257953280.0]], [[268968448.0]], [[260324224.0]], [[220420208.0]], [[278406336.0]], [[326242944.0]], [[298994656.0]], [[351935360.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
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

class PrimitiveOp_a3c23fcd52c48123761e37da451def3b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_950b961e091d5e41114ca53d32b01fb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3c23fcd52c48123761e37da451def3b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.759235858917236]], [[14.225831031799316]], [[19.027368545532227]], [[12.69971752166748]], [[13.102041244506836]], [[14.917290687561035]], [[6.757692337036133]], [[15.396809577941895]], [[16.261119842529297]], [[15.060494422912598]], [[14.572915077209473]], [[11.219926834106445]], [[12.17980670928955]], [[13.219276428222656]], [[14.74959945678711]], [[6.9188432693481445]]]], dtype='float32').reshape([1, 16, 1, 1]),
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

class PrimitiveOp_d1b129c72092749763163ef830c720b4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bceb05f64505f4fcb160e63751e0187e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1b129c72092749763163ef830c720b4
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_3ae5058f3dc0e0fc437dcd10095a1f1f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 28, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f4e1e35575fe9e778355011db836d37c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ae5058f3dc0e0fc437dcd10095a1f1f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[386696.71875]], [[238397.765625]], [[215312.25]], [[270892.8125]], [[313143.3125]], [[368320.28125]], [[282008.46875]], [[301115.34375]], [[442644.1875]], [[355851.21875]], [[369400.9375]], [[293284.0]], [[259967.015625]], [[299579.125]], [[339983.53125]], [[172553.265625]], [[202926.5]], [[392458.84375]], [[317232.59375]], [[270807.03125]], [[414629.65625]], [[271966.0]], [[191804.296875]], [[307798.25]], [[355791.0625]], [[285885.0]], [[296692.09375]], [[451372.625]]]], dtype='float32').reshape([1, 28, 1, 1]),
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

class PrimitiveOp_17ceac46a5884c18f4c085912ea22158(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 56, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d1a12663b2dc7870f2ba369796e63322(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17ceac46a5884c18f4c085912ea22158
    def get_inputs(self):
        return [
            paddle.uniform([1, 56, 1, 1], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_fe39bbabdf32e9e9a411fa519ef35865(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 60, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bf3df6ed78c958d559968285a7021950(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe39bbabdf32e9e9a411fa519ef35865
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_0ea86e06fad05036fb843a345115591f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_acbd7c8c76b5753d19119aa80c2e9994(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ea86e06fad05036fb843a345115591f
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_919bdf066a1f7d526fdd3911a086bc7e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.hardsigmoid(input_0, float('0.166667'), float('0.5'))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cd6d7d2237ee1f7d19a9fe2d84ce65dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_919bdf066a1f7d526fdd3911a086bc7e
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
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