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
class PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.prelu(input_0, input_1, 'NCHW', 'all')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6cb202c83c7c62f2f8db5726a612a3ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 512, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.25830602645874023], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_e923a2eb51f3a4b60cb39a3a52ea06b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2412177473306656], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_76960ae39f3672dcf46ca6b0ad40a4fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3150206208229065], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_e6e32269c3c3c6ffb706fb90196b8145(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.04101928323507309], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_538880291479642bc5833dd56c46c3be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.32004258036613464], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_8b64f1a7c25944b9e98f5a1e746f13ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.06682272255420685], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_8a2457d9cac843446c8c9aaa9bd1d433(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4111101031303406], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_4dd05f1f4e3f1b8b87963db5bdb6e770(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.421016126871109], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_1570deee087f01cd2d9cc79e7fc69532(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3756701350212097], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_5c081b70a9c47ea65f794eaf971d6797(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.07552824914455414], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_33b4665c0feaceed69d8c9f749ee253c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.042329683899879456], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_b3cd61797d2e011da94cb87875673998(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4937025010585785], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_ea66e8693eaf77b62bff9ac41b85a63d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.03260382264852524], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_a86638e90c3a367dc5bcd4f35258a92c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1975313276052475], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_3ba56a50986f7589c53f3eb0a94a4978(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.29452136158943176], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_65aae21b0782ab3fddf43c7bc3146738(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.21650858223438263], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_aa74ba6c2163fe25dec5a6b5dfbc43ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.38330700993537903], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_f60a7cfbd32532272c3683805d6b036b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.03535746410489082], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_04e1c76166d15ce2ea32157300f0e7f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1744578778743744], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_7dd2950740d56a607db7c3ffbb262eda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.07252088189125061], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_2136037f49fe15507e809d3cab1ad1b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.28590095043182373], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_8a138500452018d55c84c76338fc1054(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.39592576026916504], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_6b257db33353638d1ce9d93772da39aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.15063922107219696], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_2323b86dbd56c33745387ff286a12246(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.04970341548323631], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_f826758edfd9d4e167921d4a22c6e21c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.24391819536685944], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_8197c01fc17437ce83ef195c7279eaa5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3926791846752167], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_6dfc210b0583484781b44e01b2d02f8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2746190130710602], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_23fe69a21ff0d84505a75ef16c53280a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.43098723888397217], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_90279f15882de7f27de96575b29a1974(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.18460030853748322], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_8905297bb5045d4321cec8ee4908af55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3325478434562683], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_641ea5a7bc111ed971871052c8b828ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.45490434765815735], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_5b879395694040d3858b8abb91dfabe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.20832104980945587], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_4ae6ff253a584e757d4ae2fa7949dff8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2925509810447693], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_5200b0b55878b15d4476d2057732e67a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4696945548057556], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_2c93bf8af8532f468d3d6bc83cdaba95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0909171923995018], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_ee1746bd4a80e96b69b6947704ccd017(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2330029457807541], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_5a26aa6d9068eaab15dbef18f8cb12d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.32287582755088806], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_feae98da2fae4f481465c0779c8ee867(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.21458381414413452], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_bda0e46654c9c09d9d2695153041815a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.39693504571914673], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_abbfdce0d24e99d25fd95c2d477d7a1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4653010964393616], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_74ecf01b8e3e878ad4dd712bf0e81a83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4764937460422516], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_111eb1217a8d9a779ffaf02f76185459(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.24160851538181305], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_da3f5f78a691a6d7ee6ae3e35aed32ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.13344824314117432], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_1e196d41127b23b96c7546ef750921b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.451302170753479], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_a90c0b7aa369b7672478ba608e4132b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1417434811592102], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_c71f3b6dea29a148f4e62bc0012a56ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2585836350917816], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_015df8c2c4c654f30897663e960fbc6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3312363028526306], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_fa8f12ebc12fa24beff2ea1b484ffe5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.008304933086037636], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_f6a2e4b0a4b31a3baf086a5f0955f01e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.11729038506746292], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_3e869af7e2492e4a2709fb26119dfc99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3868463337421417], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_0cc399a3f689f9e16b7549ac267396a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4857531189918518], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_8bd1deaf99a8e99e67c5e4fd64da1d89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.19268809258937836], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_6309b12180f7a6f0152ef907608f1649(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.35568928718566895], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_29b8e956ba5070ebd51e6628e07f9cf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.09560475498437881], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_a7a865196a8ad11b60d01e13b09d8932(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.20159488916397095], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_4d3833aef21a6a72a6d9176b454e7bb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.302328884601593], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_8acfd9f70d55bae64ad66caa10451e34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4921485185623169], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_6c2152e094d7113e707a806145620b3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.49706968665122986], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_8b152a3b4b60decc9e67391900274cdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3677256405353546], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_38f9ee1697bbaa411f570011a28ccaf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.19390061497688293], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_a8b9c3d70bf7d9e87d543328f6d702b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.36749258637428284], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_9f6fac0c61b53c482dc4c4869a257686(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3100774884223938], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_71115d71da9912322710350440b400d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4747411906719208], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_f2a6a226be2d7ce7e78452c6b907b09e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1828082948923111], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_4f9c52481f5fa6c435b49656732f996b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.21898967027664185], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_9aee2eeb979408e8ee040b9ea077f686(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3363831341266632], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_17ef1e3c659de7668609a4afd562de95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.21174030005931854], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_b8c4b8f545503d486f6dddd88aa0ba0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2607104182243347], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_9ffcbd5f86497d697fe2ec42a9b2bdc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3564096689224243], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_2c2d5d0171db65080d7f617355d6d40e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.39578384160995483], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_487d0e9d250afa68252b1cb6df764513(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3833473324775696], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_dd1e26f90a80a31f60ee669226a47769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4186464548110962], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_e974b446f34076e21243819dd6a51471(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.20215293765068054], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_73e2b4c133ebdff811170ba312375daa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.19996754825115204], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_2f9bb13fc4e827e3c02c126c02e49644(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.03646354004740715], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_b929ff7d9f2a0927edd3d75d4d9da559(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.39400026202201843], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_9a0bada148a52f5d8498f7b9d0d17aee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.15630292892456055], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_0d21256d7a5979e91d7844cccb50e6ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4037768840789795], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_05382300f80c0f17180d9da91870081c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4567211866378784], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_253e585ee42e4ffa167d34751506d2a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.18118153512477875], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_8acb50d0a3781f9ac25beb02fa29c8c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.08639544248580933], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_91f3c07549f06495e1bdbda0b11c4d02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1534285992383957], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_e57d567149c2a1ffdb423f4b804dccaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.22142758965492249], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_b1e445b184da428abed5d38f0bc259d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.15221698582172394], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_2dc57c26e20b89d64c572b2a463121f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.27337661385536194], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_c8ee90c26bb485d532d6162fb095aebe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.45699596405029297], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_a4304631daade2517d9307355e3958c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3226501941680908], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_7a2e493e1e812e5c806393b5ed2b4900(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2616306245326996], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_54fa8dc4252b9af2a1a53161868291a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.20697444677352905], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_2ccac72fa3127928eff34546040af08a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.12197066098451614], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_9177c758ef52d90bc4e6789adfd477da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2580798864364624], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_a314059af29dddcf7a7a484c220643f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.015914862975478172], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_5fecda3132bedfb7754c65f3bca051f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b9e0fab5e81c7cf6ca7aab20167c500
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.09774675220251083], dtype='float32').reshape([1]),
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

class PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.prelu(input_0, input_1, 'NCHW', 'all')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_19b77715feaeadb643026ddce04f9295(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 512, 1024], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.392578125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_e994267a8875ab28b3cff519c34d6c52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1942138671875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_3bc15e1f150d7926e6db7e6eebd2f668(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.39013671875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_3cdb974c2c7f7bb21763e82eda7a2aea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.060699462890625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_2ed27c21dccf2a6022f96088dc72e40d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.0927734375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_0e9ecfea2eec651ae97f9ce430c26c9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1163330078125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_af0e88446f4dd0de83595181ca832652(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4755859375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_259fe5fc10663dab31e1e1d0fee178a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3828125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_7b1d4bafb1ff0ac69bbde7c9b82ad53e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.380126953125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_39474fcfa5f52f9c37e6e61ac36461f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1453857421875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_9721ace7ab12ff411f2977c29ea9b17f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.212158203125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_5be9a2bf1ce71dea12c2b029c5314986(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.384521484375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_7b04101e3c90b2cdc48953ddec491290(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4365234375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_dd2a9552ba164799215b6e296e2f5083(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.0138092041015625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_6a1c0ef5a7d19376f5676571c3574980(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.143310546875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_2646472e71a7b5b83cad8c4a8d79636b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.0229644775390625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_136c4b53a5eab6c93572db93064e3ec4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.387451171875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_295f742b61ab184fde50980562897ac6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.09564208984375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_b32e3b5349ae4261e74922efb4ccae34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.453369140625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_486ebaf6d43f2f4fa6c3133faf394e7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.446533203125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_6e65c31c0c2bfdbe444ffc32d6d37499(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.224609375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_db0e3e27bb2b75b9e5dd5344aecab9f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.026611328125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_e8720c3485323877c489d535b0b77dc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2890625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_3a0dd871cdc9fe2e9622008b103f7816(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4677734375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_fc6a85a0632cc145f49d202d6dbdfd11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.28857421875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_560005e82dd8f71045b630124d6eeced(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.251220703125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_2f9ded45cf8e39c248eb4152a3101865(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.287109375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_f4d63cff3c8daab871df17ae365da5b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.0280303955078125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_ea92a190c7b3d39a886982bc3776953a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.473876953125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_50f67122ba2abdfb4c465e85cc92847f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2259521484375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_fc01101849fa64c719ebdf04ffb061b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.292724609375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_64cb096c66c0b65d002673f7c469dea1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.05731201171875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_51e0f1a4d988884c00628bcf22344bfd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1845703125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_228e3d14772cf82c7dd9094b3279c0c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1558837890625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_6c1176a645a2ea79c98eb7cec96b863d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.19921875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_3f66f686352faf82501c60fe24df3d77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.422119140625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_f584394cad632f3d6accc656a7cc96ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.26318359375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_871da3a4b883a0cf8d2805fd07755680(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.03411865234375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_5b0f77c871c7ae61f06c4e9ba978426a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.26318359375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_84f7ad7179b606c7d31b6171940f1671(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.332763671875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_184159ca8b1bc374e567a8fc767db837(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.26904296875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_e832654b1ae316ae006ff958e5cb1e00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.468994140625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_c785eb9b8fe5fd26a3a0cb58489eeca3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.31884765625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_53efabd50ae62ed0fc069b9fef399275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1580810546875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_55059fceecea749860fc0fbadd75bfbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.0106201171875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_7956478accac92b14e4e1f9105c190e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.325927734375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_b2544ab4a06a2575d65c2cc4c25ad18b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.309814453125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_431f44e4f162a0fe9b320d8e95309003(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.0272979736328125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_7f34ddcfbbe941feb2ef0117c0cf2fbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.44921875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_9b9fe3fbf7b748704f248002a5a9118e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4072265625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_776884c5c2a5c82bdc369baf280b3bae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.452880859375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_f5cad77803c5435dd9be9f7857ab19cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2086181640625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_3cdee5f3794bc778c0efccfefdcc186d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4873046875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_7847f5bb93c36c55210426f7b091b1a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.239501953125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_bb3fca127946e93ff4d3353b16205833(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1522216796875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_c7d0051ac65d5c9529ddef4d48990817(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.351806640625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_cf02093511df2999a18632eec3ad214d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.463134765625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_6b2e4f29b43d7a98b4153a965a1c0be3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.45361328125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_c07843e50b131ce49af4dabc246dd0f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.320068359375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_37deb256141bafe82645b0e3660c3e7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2958984375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_4b4151f44a280aacb727a29dd85acefd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.0887451171875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_78e4ee59978bff5f7f59e4fb0e9cad34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.32080078125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_7e2cb0363736572dc18e113569329513(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.177001953125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_0ccb8f486ce2f8349f5570087fd2ae03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1798095703125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_6dac86bf15e5eb82e2c04978da1a6048(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2257080078125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_ca2ebc2f23c877d454dd85a0f2aae058(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.10626220703125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_08f2ba341e0d7965160d9804beaf70e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3837890625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_213c7433984f46e652eb4d2c56d2b952(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2254638671875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_24fddd5920e49ca0c8204bbdb47991ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.440185546875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_30aab3f0b68984be0c588f0808219256(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.428466796875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_59e539e2a6d1d2cc32d1cd2b86a261a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1844482421875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_8cb4d7d26827d7bec1a4950a32754cf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.43310546875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_8ae120baac57395bd1a838cf83d4c59f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.280517578125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_ac737cef5846f35cfa261fe837d3bfa3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.326171875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_32964df952bad0049365c7df140e8ab2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1676025390625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_b7dc90c87ddde114cde696411c44851f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.15380859375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_d9f591ea9ed4316007c77b6fc2ee32bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1282958984375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_4a33eecc75e0bf3e74a755830646cce0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.22412109375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_53142730682d02d674c6517c4ad317f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.135009765625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_880d886f7310a9a8ac8e0327c508e357(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.0223846435546875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_071a419513cf4146eb4c7d8809d1d7c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.45849609375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_622b9833865314e24e08a9e36c247d61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.15478515625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_e45e7d5c379aff59399533dcaf8b1008(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.0003876686096191406], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_26921cc4acbe6700cd26457defb03438(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.36474609375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_82b10be25d8f4770e333aef9ff8d80ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.406005859375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_1fe66bdebf5ce1e02a945312a76d5324(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.16064453125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_6a5f72691555f411f085c75614fbf0c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.33544921875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_918cc82f6607d68f37bce92ad433a3b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.42529296875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_fbbb8139cd34897cdf37b3586b887d37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.383544921875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_a203abd9b3962309063615f57f1a94bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.472412109375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_f0559fa47551dafd77b93d43b89f5cd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1768798828125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_7b30f4f80f0f8d2088859505a116b7b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.41748046875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_4a16bca6475b15969917db90e3cffeca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aeecfdc011376b4480dd2488c5a6b61
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1517333984375], dtype='float16').reshape([1]),
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

class PrimitiveOp_eb949b3bd53e65cf2c8e248046d85a00(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.prelu(input_0, input_1, 'NCHW', 'all')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 16, 512, 1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_789ad9adf4a8a5ba57e37ca78233634e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb949b3bd53e65cf2c8e248046d85a00
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 512, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.25830602645874023], dtype='float32').reshape([1]),
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

class PrimitiveOp_492f2df027d2ee830132caa0f27520aa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.prelu(input_0, input_1, 'NCHW', 'all')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 256, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_89850c55ffa3b23058704ebeffd6d889(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_492f2df027d2ee830132caa0f27520aa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2412177473306656], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_2074c2df9e8df5c82f40b962eaa82be2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_492f2df027d2ee830132caa0f27520aa
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3150206208229065], dtype='float32').reshape([1]),
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

class PrimitiveOp_1979ef080e7abea654a076844d5b1bae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.prelu(input_0, input_1, 'NCHW', 'all')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 256, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_38593573d3b1242f9b218271ed486cdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1979ef080e7abea654a076844d5b1bae
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.04101928323507309], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_f22b4873fa101c16e14872e5800ae354(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1979ef080e7abea654a076844d5b1bae
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.32004258036613464], dtype='float32').reshape([1]),
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

class PrimitiveOp_9ad112dec18b1c9346710e12dda0a936(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.prelu(input_0, input_1, 'NCHW', 'all')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 16, 256, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e2e40c86925b3f55fd8b51e310a8028a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ad112dec18b1c9346710e12dda0a936
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.06682272255420685], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_50f9d306630be2003c999b63592b99cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ad112dec18b1c9346710e12dda0a936
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4111101031303406], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_05a9ee136d9ff8d78216b6e43d957567(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1979ef080e7abea654a076844d5b1bae
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.421016126871109], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_a5d11bb896fb964f834f5c79d32efc98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1979ef080e7abea654a076844d5b1bae
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3756701350212097], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_5a734fd1cf1808e17385d4f83863f435(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ad112dec18b1c9346710e12dda0a936
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.07552824914455414], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_225d90b3ce86344765657da35dc0b1d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ad112dec18b1c9346710e12dda0a936
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.042329683899879456], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_3dd74447b5d7bce5669cd0269f8c6eba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1979ef080e7abea654a076844d5b1bae
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4937025010585785], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_98a113fd1c7d350a4454ee0e33e85c99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1979ef080e7abea654a076844d5b1bae
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.03260382264852524], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_274ffb1b48a0a24736a4efcb1f27d698(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ad112dec18b1c9346710e12dda0a936
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1975313276052475], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_6010c7a6cce47d62178b2e58335c4d23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ad112dec18b1c9346710e12dda0a936
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.29452136158943176], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_3f932e43c57582f9c4ce640d198d9edb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1979ef080e7abea654a076844d5b1bae
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.21650858223438263], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_864a9a1c455c43ddb9814bbc7af96db7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1979ef080e7abea654a076844d5b1bae
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.38330700993537903], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_0f8869a53008349be70cf6d8d8e84976(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ad112dec18b1c9346710e12dda0a936
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.03535746410489082], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_c7e114014e047da7334b532764fb13a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ad112dec18b1c9346710e12dda0a936
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1744578778743744], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_f7f6f65e7bcff494d84de6f68fcf2028(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1979ef080e7abea654a076844d5b1bae
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.07252088189125061], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_77c7fc893c25895e79b8f7cb888be8f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1979ef080e7abea654a076844d5b1bae
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.28590095043182373], dtype='float32').reshape([1]),
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

class PrimitiveOp_50e12a4b69ba863d0af1e8f8be578ef5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.prelu(input_0, input_1, 'NCHW', 'all')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 16, 128, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e0bebed4a2562d950f2bf69fa1c495fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50e12a4b69ba863d0af1e8f8be578ef5
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.39592576026916504], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_03aa2c4e95764b96c670917fed35db23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50e12a4b69ba863d0af1e8f8be578ef5
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.15063922107219696], dtype='float32').reshape([1]),
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

class PrimitiveOp_9441b45d7aabdc85763997387e3655f7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.prelu(input_0, input_1, 'NCHW', 'all')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 128, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e42c387f5e0795b7d51f339c80222be6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.04970341548323631], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_fbfeb9af42c6e163247adf2c269779bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.24391819536685944], dtype='float32').reshape([1]),
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

class PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.prelu(input_0, input_1, 'NCHW', 'all')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 32, 128, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_488162b61c3a25c9e4f9ad857b4c45f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3926791846752167], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_e365c551ae383dc91df85de4575e496b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2746190130710602], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_1733067db6e79c60872f4a2a3af83705(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.43098723888397217], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_c1eefe8abcf9e5738bbd527fdb741552(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.18460030853748322], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_a4513303dc6dbed362e163560ebda3e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3325478434562683], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_2da948791eb96b16b7fd1ccd7ec95a7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.45490434765815735], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_ca397b1463fc3db308acc5cb304a5f58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.20832104980945587], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_7ff781e9c9c8fa233c01ea29fae3d591(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2925509810447693], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_f5df35a17335c186d21d5876edea0323(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4696945548057556], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_c045b7603693436f25a804c5facdcb11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0909171923995018], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_e1ff3932a3186c07d61cdaece91ac9bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2330029457807541], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_42c32c084cb1c716479d09fa98f6c755(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.32287582755088806], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_ba775248ce320825ae08dacb1809cb5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.21458381414413452], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_758f033ba90c408a396c92f49c222ecf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.39693504571914673], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_9adb03afa64dcebbb996cfb94ed1847e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4653010964393616], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_dfbb6467524dc903334ef20b2e6e0d16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4764937460422516], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_ad07991752a57e771f31e62b017e7341(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.24160851538181305], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_08d9a803a8d9ea31e272ae8af6284f25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.13344824314117432], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_76a2626d48ba8b88826d3571bb7351a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.451302170753479], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_dbd76d744918d31f1cbbb1dd9fbf3669(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1417434811592102], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_b437406019ff33b45df1d477f359a295(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2585836350917816], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_13f07045d7936220e6468882c11bb03c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3312363028526306], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_5b973588f4bc1a08456bd402fc59a7c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.008304933086037636], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_92d9582ef00a766663cc94c838107594(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.11729038506746292], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_09aa70d405e0f66ff5cc4d11457b8f76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3868463337421417], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_cb4fa6a003c506abd739ade018b73026(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4857531189918518], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_12d6b744b12c94fccb348761f8aae3cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.19268809258937836], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_9dfd94dd75ca29b077812a103991c5bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.35568928718566895], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_4c716fbba7a130c5405ca5425f7fb3de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.09560475498437881], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_9870564af03c08164af597a4dce698fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.20159488916397095], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_1be755661b33bb6afca82afc0736433a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.302328884601593], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_cff1991f3286ab195ed6a55cd0283f92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4921485185623169], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_27366ef280df3314cf8d87d8487321fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.49706968665122986], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_a2e1eb027143d55078c29ad0d4c792a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3677256405353546], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_5cdd0f007390203c3affc11124b3ff7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.19390061497688293], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_7abbc6740efe587d0b996f2fa948bfac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.36749258637428284], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_e80dbe15693c72c533f5f5912788c9d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3100774884223938], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_0539b44de9053fd1164b9c60b7f78d16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4747411906719208], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_f9e8a986c5d77da8f16d7f55798bb846(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1828082948923111], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_1581a3ec40893b662404c335c3baee30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.21898967027664185], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_5842e217da801a533bea6524db080407(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3363831341266632], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_0a2967484ea9cf2767639836f4eb9285(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.21174030005931854], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_429b7f2ad28af7271b2477d23826e217(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2607104182243347], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_e97c8a0e186a71745842f82a6481a3fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3564096689224243], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_563136d342785b0867c057134ebf85a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.39578384160995483], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_641ca8869d4d92864d7a1bf45409562b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3833473324775696], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_e5aab30585bc2de8304bc00a8149f0bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4186464548110962], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_2b68e4d3be33920046a1b309a12066eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.20215293765068054], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_a7b7bb3b68d6af05bc9b8fbb14a3dd93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.19996754825115204], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_6fd7b2031e382d612658643b26611d67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.03646354004740715], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_6658007af7cb5001332093aae1b38d31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.39400026202201843], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_90f42a16133c3ade1478c77f2f9934cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.15630292892456055], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_5eb79ae4fe593af16592530731cc406e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4037768840789795], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_45851f40b4c7df3b1aa1f06c44130270(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4567211866378784], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_7e0f4ca6e47eb18d92bc1579ff7c15c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.18118153512477875], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_2bd2d806f3af9f1a41e9ad8596945734(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.08639544248580933], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_3304c2fd715a3a90235bc90d5e4d5de9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1534285992383957], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_ddc18d7a672adf6b098f686bda88fb0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.22142758965492249], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_bcb13a471e7cf5b9c0f5296d8c7d26dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.15221698582172394], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_cbe8e3975226c3addc6db789bbbaa9f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.27337661385536194], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_993f8f906ee159ac4839b1c273cb5810(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.45699596405029297], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_916aacdd113cb2cac55e047c517a6506(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3226501941680908], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_8d98f8a0570c16487e04221376330b34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2616306245326996], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_8fdbd6831c3a766ffa0741ae42f9e8e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.20697444677352905], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_6d695549ef8c1471cbe1042b72717d6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.12197066098451614], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_02c94411d13823cced45bed39562ae75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7e9dd4e196cfb19953611fb5a6666c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2580798864364624], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_cccf2518bfc3ffecbe39faaf63649bd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.015914862975478172], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_efad523b11618814b9bbfc287a037f3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9441b45d7aabdc85763997387e3655f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.09774675220251083], dtype='float32').reshape([1]),
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

class PrimitiveOp_1ec43a1f647a5d2dc0df55a48b9fa06b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.prelu(input_0, input_1, 'NCHW', 'all')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 16, 512, 1024], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_860b0ebe97092ca7ba4057449c83c99c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ec43a1f647a5d2dc0df55a48b9fa06b
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 512, 1024], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.392578125], dtype='float16').reshape([1]),
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

class PrimitiveOp_124bebe20410c3f1eaed791018184aec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.prelu(input_0, input_1, 'NCHW', 'all')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 4, 256, 512], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f9bb7e378d6e37326471295c309136ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_124bebe20410c3f1eaed791018184aec
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1942138671875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_f74fc57d98c973ef97632521dac6eef3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_124bebe20410c3f1eaed791018184aec
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.39013671875], dtype='float16').reshape([1]),
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

class PrimitiveOp_a962c3f203ba279f861e0f902a24edd7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.prelu(input_0, input_1, 'NCHW', 'all')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 256, 512], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_110e980b2b104b9ecb637b6257a67e2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a962c3f203ba279f861e0f902a24edd7
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.060699462890625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_e49232ae12c3cf83f2113a5c941763a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a962c3f203ba279f861e0f902a24edd7
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.0927734375], dtype='float16').reshape([1]),
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

class PrimitiveOp_dff9a9fa28c131ae8b3f9b73e8f22f6f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.prelu(input_0, input_1, 'NCHW', 'all')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 16, 256, 512], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_da3d0f64d7308d8d32213cbc38dbe466(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dff9a9fa28c131ae8b3f9b73e8f22f6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1163330078125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_e9c9228595bd5cda901b9dd4ffb1baed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dff9a9fa28c131ae8b3f9b73e8f22f6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4755859375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_0d58148e1f81950abf7dbf6f56801230(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a962c3f203ba279f861e0f902a24edd7
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3828125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_2ce8155eb3f9364993dc9fd608381eb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a962c3f203ba279f861e0f902a24edd7
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.380126953125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_ad1f1229729e188c1eef57a6f1bbef09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dff9a9fa28c131ae8b3f9b73e8f22f6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1453857421875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_556589a683f95fccfd57299017eb0b46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dff9a9fa28c131ae8b3f9b73e8f22f6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.212158203125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_6a0d0660adac99f9b7e941de3298ccdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a962c3f203ba279f861e0f902a24edd7
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.384521484375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_33cd5fd84cd07fe216a15cccac057dff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a962c3f203ba279f861e0f902a24edd7
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4365234375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_e117b8ac7008c498102e0a10fcfee59b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dff9a9fa28c131ae8b3f9b73e8f22f6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.0138092041015625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_04d484f59e9bcd2499f4329089e9b232(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dff9a9fa28c131ae8b3f9b73e8f22f6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.143310546875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_bb79b53b36fbc7ef7c566e48aaddca28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a962c3f203ba279f861e0f902a24edd7
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.0229644775390625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_2c5fdbdd0018c8544ca2f6150a982f7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a962c3f203ba279f861e0f902a24edd7
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.387451171875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_66d684d6f443d607218a03e1333d71cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dff9a9fa28c131ae8b3f9b73e8f22f6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.09564208984375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_210121c63622bb0cb4f45e4db888d102(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dff9a9fa28c131ae8b3f9b73e8f22f6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.453369140625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_0e8571a19e8ef3581e51bd85b427bc53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a962c3f203ba279f861e0f902a24edd7
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.446533203125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_445251c30354256d3839b76e7fadc091(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a962c3f203ba279f861e0f902a24edd7
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.224609375], dtype='float16').reshape([1]),
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

class PrimitiveOp_110fdf1accdef04f84eb4771cfda19d6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.prelu(input_0, input_1, 'NCHW', 'all')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 16, 128, 256], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3e7024689cd17e5c1f2a1a16d17e4635(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_110fdf1accdef04f84eb4771cfda19d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.026611328125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_4e64ddfc69185051bbd70d920ca16d10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_110fdf1accdef04f84eb4771cfda19d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2890625], dtype='float16').reshape([1]),
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

class PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.prelu(input_0, input_1, 'NCHW', 'all')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 128, 256], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2463f8d2c7ed4b58b4d2e36e2d69a882(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4677734375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_001fd41eced5c450e6ae3ad9c0cb42c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.28857421875], dtype='float16').reshape([1]),
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

class PrimitiveOp_47b513ca6248a9617946515ab55bcdef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.prelu(input_0, input_1, 'NCHW', 'all')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 32, 128, 256], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_641e0161739ed75d2a32694dc24a27f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.251220703125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_538a9abc4de9534390fef9966d8200c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.287109375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_117c9593bed9379f2aa18c6eac000cac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.0280303955078125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_efc506115dd162ed36387b34022257aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.473876953125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_c14cc068bd05dea2101864aa84c11585(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2259521484375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_5dffd9dc17f3d4ec647f45b9ffee44bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.292724609375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_67451e8cd4fb4ba9caa83edf73addcf5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.05731201171875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_1cf83e1a212c9b5685c426bcf7b02a2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1845703125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_a9d805bd976bdeaea7afc4943dc11690(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1558837890625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_eebb7d1d115ce7402252bd9e76d1d004(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.19921875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_396a18a21fa435a4dd2fae7838731bbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.422119140625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_20358d349e36bc17338b8513a99a6030(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.26318359375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_3e00678890885cfed7abe7f5f5d0c475(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.03411865234375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_ab6a76c85b06d900303b391bbb07325c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.26318359375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_3ac778a39539f645308c2e64e7f73b59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.332763671875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_8811ec7b43055300d6d604467fe36b71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.26904296875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_c27c7ddfc83fe3b9ee11e73df38844c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.468994140625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_a658bba7af8b15ebf97e185004f5a0b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.31884765625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_f5991c5e1afe98331b402d521138829e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1580810546875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_a4c3373d94fe6dcde310b3071c8126b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.0106201171875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_8ffe2263a7a52be1b529face88fbb02e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.325927734375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_d1ed28e390b34ff65a1d3ee2127fa19c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.309814453125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_064551a1ece93ce7deb1b41315a5cc94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.0272979736328125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_0ab9c331ce25bf7a9e346a0f99bd997c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.44921875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_55201aabc495080e4b923121f2bee488(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4072265625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_9b19bbe4778e0a4e2901fe5ccf061d97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.452880859375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_935075e64f3ae880c85fc2e5ff9faae1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2086181640625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_bbf3a4f9f7920ae09f6fdc5ef4b04f11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4873046875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_064c159b0a42504bc903a563fec05c6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.239501953125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_fca67ae6f470f79a3451d8a6d66819ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1522216796875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_830901b08a2b5407532b4dff42797639(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.351806640625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_c845d319c6f00971838503c69a26964b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.463134765625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_9df83c92988900b29f57326a0ba254be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.45361328125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_2cc5af67b6d20f92bf756be9b9b6eeef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.320068359375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_49610aa4f5b8aa54cdc59c44e7c78b7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2958984375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_a88dd217db64a94e6bf476b61ffa64cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.0887451171875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_82c409b0c2676fe79976e256df96bc1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.32080078125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_0ee609f93cd13eb54dbe3c002b80ff1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.177001953125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_45ed337c5d1297f16af92024a339be82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1798095703125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_98f0327142bbf2793de852a8439364a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2257080078125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_0e9e6cf2f917547b7a3439739a196ced(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.10626220703125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_35ab69fbcd7532b578a48cd2fd6d80f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3837890625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_1fc7abea897f20dd1b54974231b799e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2254638671875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_8f5374792b3e336dbdded5e54bfbf113(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.440185546875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_a4d4157e23c83ed266c3390986c7e5e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.428466796875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_da7f299c1c79faff50507e353174d3f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1844482421875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_b108cc417764ffeb85364f64825acad9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.43310546875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_8c8d62dabd4a618ff4c0f9553b16b7d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.280517578125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_139f858c212dc50da0d34a34d083d2e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.326171875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_54816f52158762c62fb8f7c8748908e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1676025390625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_63f7d5d96d34a5683c0a1436957af28a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.15380859375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_1c52fda5178c8772367adfa1ba69f6dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1282958984375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_07d93f2b19c1d4ea603dff35d1710752(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.22412109375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_78fe2ae22a3288e4f4ec2bb09781bd31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.135009765625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_377c5fa6be8c3e63837a78162b8c6ed8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.0223846435546875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_ac724e881301ef9f775bbbc18d790c9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.45849609375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_96b582c09d4ce371e70cc87cc427d3f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.15478515625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_6adce7b5ce5918bca1bc1cfe610849a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.0003876686096191406], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_8fcfdac937b8a2dbf005bfafef69bf2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.36474609375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_3444f2a6a7e9717b75640a385a267abf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.406005859375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_26b977361ced74278e36519239a47205(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.16064453125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_52486a6d6a58f51bab4cf5984c1f80a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.33544921875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_887a8da6fcf56c96f9acbe142bf83458(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.42529296875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_ccb6d4a5d73878c0e4cf8833dd74c924(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.383544921875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_e67ab22c800ac5a467004e5eadce7d3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.472412109375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_e5c90dad459f74856aecbf41f3f8305f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47b513ca6248a9617946515ab55bcdef
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1768798828125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_921fd68f710ff13037e141f9345b949c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.41748046875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_4c6b4ffc7c53fe8aebb476c85936785a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e84934ef4ecf371081d148fed1ea3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1517333984375], dtype='float16').reshape([1]),
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