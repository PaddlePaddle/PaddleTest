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
counter = itertools.count()
class PrimitiveOp_bcc827b79bd840d26c0466333bfa67de(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_04a7fa6710f3bd8e1c69c804da35ea09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 320], dtype='float16', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0308ebe4c5701543f2b94ee28f1e5043(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9587c7d08393a7ca4b3e9d88f1c980c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.007542538922280073, 0.2589527666568756, 0.487557977437973, 0.29581987857818604, 0.39868220686912537, 0.12857985496520996, 0.17470674216747284, 0.21117405593395233, 0.13679154217243195, 0.1589573174715042, 0.3821004033088684, 0.4808706045150757, 0.004323906730860472, 0.26371440291404724, 0.45760050415992737, 0.38391822576522827, 0.18062378466129303, 0.13580891489982605, 0.06980408728122711, 0.3521064817905426, 0.463852196931839, 0.033546581864356995, 0.12499123066663742, 0.3425736427307129], dtype='float32').reshape([24]),
            paddle.to_tensor([0.1796165406703949, 0.04581243172287941, 0.2864757776260376, 0.11248375475406647, 0.2057366818189621, 0.37921521067619324, 0.43359479308128357, 0.18197079002857208, 0.008127328008413315, 0.20194712281227112, 0.23519830405712128, 0.047192152589559555, 0.34459710121154785, 0.33789950609207153, 0.33767247200012207, 0.4074074923992157, 0.47153669595718384, 0.04866562783718109, 0.4100349545478821, 0.3305538296699524, 0.15910589694976807, 0.4810023009777069, 0.43099188804626465, 0.06479601562023163], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0e91964e6f3a7600c14098554165fc98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 197, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_b40192c1e65b36dea8f489ebf100b854(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_65d079b8ad81a9662942ac4f40fd954a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40192c1e65b36dea8f489ebf100b854
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_04ebe9bb6ba15cb7c6a3968b62c01530(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 512], dtype='float16', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_a254031597d8288871b0422f0fa8e363(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fc2895c63d7b792d9e5951fceda08c59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a254031597d8288871b0422f0fa8e363
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 320], dtype='float16', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_687cd369fdd3cf9dc499093ab2d492c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a1bf8870d2da577eda88381bc273949f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5e10c294f1eab30dd0c9f06b8dcdbe07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 197, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4107a7be698cf39c1a9594518d36a877(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a254031597d8288871b0422f0fa8e363
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 384], dtype='float16', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_83342861c7ffaebc9c9a3ab08c6494f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 384], dtype='float16', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cbdf61b0c4ffebf66ef7bb9e85d2635a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 25, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ac186de53eee3328cf2e88dcc7562c3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1938f81f7e9ef403b9b7ee161b1f9c66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a254031597d8288871b0422f0fa8e363
    def get_inputs(self):
        return [
            paddle.uniform([1, 197, 768], dtype='float16', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4f00206646f3be2f4de3821744d7a979(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 26, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d90eea07f23b57be5f9ff68a6d62521f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40192c1e65b36dea8f489ebf100b854
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6ba2c6e94bc4f3d0fde967a17a21d6eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 25, 512], dtype='float16', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_234e8939b6f7b81d981d178debeb11b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 128], dtype='float16', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_068181e1320356cb5f55f144f7a24875(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_168566e56e904df1298e659419eb0b8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([4, 256, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2aca2a88b69d22f5737a5b316c167916(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 384], dtype='float16', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4c697ae9d77861bf2a38924914465eac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ddccf71a9d24de2712006dd32014aaa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a254031597d8288871b0422f0fa8e363
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 768], dtype='float16', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f9b3e9f3d94923916bcb34376cd88756(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 512], dtype='float16', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9a767c8c1de3063eef200012e67eb0a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40192c1e65b36dea8f489ebf100b854
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6f20aa71779fde912d5eec52102c9eb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40192c1e65b36dea8f489ebf100b854
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_405a41dafda22494a682d582ffe6be4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 256], dtype='float16', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e0f75b9c8daa9da2cbf957d0f4258fc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40192c1e65b36dea8f489ebf100b854
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f4f4ac0b1548bd48471dd6654bec7cd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40192c1e65b36dea8f489ebf100b854
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ee791e0d496884ee4e79377606a60288(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6b0bbca2b4b85b255a35f1f5d169e4f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 197, 768], dtype='float16', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2173fe57189fce78b472d5539fd0b18d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_983cf177e7203a222c63fa668ee59f49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 784, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e049124f4268ecdb6e87fc6abfe97675(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 784, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9289ae0ebcd444d1ef3d27a94f344315(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a254031597d8288871b0422f0fa8e363
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 768], dtype='float16', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_59e39f5d3004abcdda5546b4d19bde93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 9216, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_13133281d4f1b869fd71350db0059266(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a254031597d8288871b0422f0fa8e363
    def get_inputs(self):
        return [
            paddle.uniform([1, 3136, 96], dtype='float16', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c479bd7d8a3ea86b7fc351570fa13503(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a254031597d8288871b0422f0fa8e363
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9904ff3fc9ed46a9364a1a50e3fa8c27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0b221a77726a2cfc6791c27b8d77cec5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a254031597d8288871b0422f0fa8e363
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 512], dtype='float16', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c179800a67b5fc50227b0b5d34e649fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1944912225008011, 0.3826028108596802, 0.16157162189483643, 0.3195856809616089, 0.3475199341773987, 0.1882621943950653, 0.08792823553085327, 0.18206702172756195, 0.10100555419921875, 0.22334177792072296, 0.12090813368558884, 0.048660337924957275, 0.28477728366851807, 0.41904664039611816, 0.4214280843734741, 0.44576936960220337, 0.23108689486980438, 0.16022145748138428, 0.3058986961841583, 0.18708254396915436, 0.3134671747684479, 0.4069231152534485, 0.2430054396390915, 0.2621396780014038], dtype='float32').reshape([24]),
            paddle.to_tensor([0.32334986329078674, 0.3996644616127014, 0.49635884165763855, 0.47662168741226196, 0.4875451326370239, 0.07987800985574722, 0.2242894321680069, 0.30241909623146057, 0.02477540262043476, 0.24675896763801575, 0.33077937364578247, 0.03814496845006943, 0.49355798959732056, 0.45218390226364136, 0.23609976470470428, 0.23890382051467896, 0.2221267819404602, 0.4675355851650238, 0.15177153050899506, 0.1619296371936798, 0.2415061891078949, 0.3584093153476715, 0.17826193571090698, 0.26559266448020935], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6c0a50e1d9d3ea979b57b3641ef4d88a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40192c1e65b36dea8f489ebf100b854
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_578b5b0d43c98ead33a2ccab23e286af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 784, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bcb20e3316390a97eb14252c47224ab3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40192c1e65b36dea8f489ebf100b854
    def get_inputs(self):
        return [
            paddle.uniform([1, 784, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3950336b1c0088f1d835dc0a2fbee9e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40192c1e65b36dea8f489ebf100b854
    def get_inputs(self):
        return [
            paddle.uniform([1, 197, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0d7b41405c7f3fcf758103340f8cb211(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 784, 192], dtype='float16', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f23ec4eafbf7a3258de35eb8f79baf78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40192c1e65b36dea8f489ebf100b854
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8673a0b0daba627cbc078c5691d903e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a254031597d8288871b0422f0fa8e363
    def get_inputs(self):
        return [
            paddle.uniform([1, 784, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_84d05576fcd2882cba2d842114f48766(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 320], dtype='float16', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_79187b3b158eb1dc09e593c5b19f4178(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([4, 16, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_667997ee8fc6261db030cd15ba679d06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a254031597d8288871b0422f0fa8e363
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 160], dtype='float16', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_effd3743acd3301ecf95bd7e84e234e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 197, 384], dtype='float16', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_460f3318fdbb8ee5ebecb7761d45ded2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 9216, 128], dtype='float16', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_35d0326978897cc39a7b689e3d7faf2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 26, 512], dtype='float16', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_50399217fdf7209a51e0cecf5214afa0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a254031597d8288871b0422f0fa8e363
    def get_inputs(self):
        return [
            paddle.uniform([1, 3136, 32], dtype='float16', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ecaedb8b0672bcddd8a6a6576485e792(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a254031597d8288871b0422f0fa8e363
    def get_inputs(self):
        return [
            paddle.uniform([1, 784, 192], dtype='float16', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b862d93ee38e3c975ef018336d38328c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40192c1e65b36dea8f489ebf100b854
    def get_inputs(self):
        return [
            paddle.uniform([1, 784, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_420702d25a9d162c1a1eceeaf7845bef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 3136, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2c1a8c98e0d4004ae4b090f101f29602(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.377737432718277, 0.18530283868312836, 0.14323432743549347, 0.48664844036102295, 0.010780871845781803, 0.4611663222312927, 0.4772086441516876, 0.37828487157821655, 0.1962117701768875, 0.3811570703983307, 0.4023241102695465, 0.28489604592323303, 0.16372263431549072, 0.44700175523757935, 0.25237423181533813, 0.4096340537071228, 0.0903124287724495, 0.04274021461606026, 0.3045683801174164, 0.38907498121261597, 0.0076731350272893906, 0.40988290309906006, 0.26395905017852783, 0.42480334639549255], dtype='float32').reshape([24]),
            paddle.to_tensor([0.042607251554727554, 0.2292732447385788, 0.2175278216600418, 0.4017587900161743, 0.21708501875400543, 0.11139581352472305, 0.12987275421619415, 0.01210138387978077, 0.3472575545310974, 0.3450187146663666, 0.02834395132958889, 0.4251275360584259, 0.2360263168811798, 0.465384304523468, 0.13669751584529877, 0.08593716472387314, 0.023051662370562553, 0.33587512373924255, 0.09113654494285583, 0.3745057284832001, 0.3440038859844208, 0.2501061260700226, 0.15599140524864197, 0.1263386309146881], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b73dc518ad4501b4888e62664d439fa5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.38617104291915894, 0.11572497338056564, 0.38189390301704407, 0.16020172834396362, 0.29765796661376953, 0.017672955989837646, 0.03630007058382034, 0.34577760100364685, 0.23633930087089539, 0.294071227312088, 0.4245051443576813, 0.43033868074417114, 0.031015468761324883, 0.2822730243206024, 0.3299134075641632, 0.15210506319999695, 0.49575263261795044, 0.19125495851039886, 0.41863080859184265, 0.3343929052352905, 0.2654281556606293, 0.18332819640636444, 0.1981712132692337, 0.33713534474372864], dtype='float32').reshape([24]),
            paddle.to_tensor([0.316145122051239, 0.27618157863616943, 0.3883530795574188, 0.36047354340553284, 0.4580325484275818, 0.27429041266441345, 0.31387272477149963, 0.31366121768951416, 0.17350850999355316, 0.37955591082572937, 0.03064313903450966, 0.47621357440948486, 0.4102761447429657, 0.3021687865257263, 0.2562055289745331, 0.06368464231491089, 0.3903196156024933, 0.2787001132965088, 0.219029039144516, 0.048097074031829834, 0.2542375922203064, 0.3385213017463684, 0.46876662969589233, 0.019238121807575226], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7ed4bf24daf2d494dfdf343eb38b23e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a254031597d8288871b0422f0fa8e363
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 128], dtype='float16', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6a425d05072666546bc66760fa925746(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.39341840147972107, 0.46687382459640503, 0.10935653001070023, 0.4551464021205902, 0.40478137135505676, 0.3409687578678131, 0.34628233313560486, 0.4930148124694824, 0.4607038199901581, 0.323678582906723, 0.34648799896240234, 0.2895970642566681, 0.35318222641944885, 0.4281359612941742, 0.10394617170095444, 0.2694813013076782, 0.017427900806069374, 0.25487905740737915, 0.292839914560318, 0.4755801856517792, 0.18176168203353882, 0.4752233624458313, 0.21155650913715363, 0.3634304404258728], dtype='float32').reshape([24]),
            paddle.to_tensor([0.4130283296108246, 0.15411455929279327, 0.45077526569366455, 0.45445412397384644, 0.15674017369747162, 0.48320630192756653, 0.3298322558403015, 0.4855938255786896, 0.4683642089366913, 0.4628073573112488, 0.10784704238176346, 0.19944587349891663, 0.3011212646961212, 0.1455358862876892, 0.2366676777601242, 0.3454083204269409, 0.1815304011106491, 0.3239162862300873, 0.2077907770872116, 0.24845445156097412, 0.07662981748580933, 0.4485117197036743, 0.3096381723880768, 0.17961277067661285], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e8bc50355e2b9c1d33e2e76635beb9cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_35f8f728f303c9ce68658f4aa6b02a49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40192c1e65b36dea8f489ebf100b854
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bf5a758259e90d56155636375348045c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.25697630643844604, 0.2616826891899109, 0.36537230014801025, 0.13864648342132568, 0.04111538827419281, 0.20468221604824066, 0.3774813413619995, 0.30760225653648376, 0.3268921971321106, 0.16162121295928955, 0.193632110953331, 0.07153741270303726, 0.05428795889019966, 0.3097993731498718, 0.23834088444709778, 0.37003788352012634, 0.04955434054136276, 0.3636513650417328, 0.48130127787590027, 0.425507515668869, 0.07777732610702515, 0.09686791151762009, 0.47500738501548767, 0.17763380706310272], dtype='float32').reshape([24]),
            paddle.to_tensor([0.03983965516090393, 0.011194800958037376, 0.33525604009628296, 0.4724425971508026, 0.22037267684936523, 0.22979885339736938, 0.4378463923931122, 0.3930085599422455, 0.2661578953266144, 0.06824540346860886, 0.30339109897613525, 0.2867439389228821, 0.060707103461027145, 0.3062986433506012, 0.29465994238853455, 0.46912169456481934, 0.025963183492422104, 0.25091734528541565, 0.042882289737463, 0.4332965910434723, 0.1658511757850647, 0.15828007459640503, 0.11682777106761932, 0.12158512324094772], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dbd3147bf3edf6f2132a409aeb9d6a3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e4d168ceffed1560bd10c7a01d2af642(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40192c1e65b36dea8f489ebf100b854
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7f8977e56d7d0bd5ed4b3e0ded365bc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8eae3b4d7191255835c177c1f57e3297(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d1d4c36a4d24492949620b1853e8ebce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2218901515007019, 0.36188772320747375, 0.04948639124631882, 0.47907280921936035, 0.19810333847999573, 0.2595363259315491, 0.2106567919254303, 0.42106103897094727, 0.13477306067943573, 0.13146597146987915, 0.08650773018598557, 0.07896021008491516, 0.1654050201177597, 0.08941850066184998, 0.09084242582321167, 0.18809713423252106, 0.46906110644340515, 0.14247792959213257, 0.07266838103532791, 0.4778778851032257, 0.16100482642650604, 0.3071631193161011, 0.30422717332839966, 0.019381634891033173], dtype='float32').reshape([24]),
            paddle.to_tensor([0.003561025485396385, 0.46045517921447754, 0.21475692093372345, 0.010452022776007652, 0.3242853879928589, 0.2562505602836609, 0.33443042635917664, 0.48823460936546326, 0.48607632517814636, 0.33735978603363037, 0.4287622570991516, 0.28406837582588196, 0.24270394444465637, 0.14426647126674652, 0.10508942604064941, 0.10875041037797928, 0.02983558550477028, 0.4991454482078552, 0.1449068933725357, 0.2402506321668625, 0.3209373354911804, 0.06510025262832642, 0.0939621850848198, 0.3456525504589081], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9f2c7fdd899a5d5dafcc25c695f884df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a254031597d8288871b0422f0fa8e363
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 256], dtype='float16', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6aadf27a816de5badeb9c95e1f0273c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([4, 64, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1b30ab7546c312d8bc55da537e3b8b3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.21724389493465424, 0.21386022865772247, 0.33139729499816895, 0.42786169052124023, 0.23576483130455017, 0.39199674129486084, 0.4420982897281647, 0.18765150010585785, 0.40616729855537415, 0.2982950210571289, 0.030321791768074036, 0.023114165291190147, 0.03075837902724743, 0.3489494025707245, 0.31738221645355225, 0.3437778055667877, 0.3625026345252991, 0.42859336733818054, 0.2719637453556061, 0.17685560882091522, 0.21495655179023743, 0.1715925931930542, 0.433343768119812, 0.4940958321094513], dtype='float32').reshape([24]),
            paddle.to_tensor([0.12360720336437225, 0.47353073954582214, 0.2214878648519516, 0.16142845153808594, 0.14972840249538422, 0.03269874304533005, 0.3545283377170563, 0.4615774154663086, 0.012213196605443954, 0.012177485972642899, 0.07326292246580124, 0.45970240235328674, 0.04811980947852135, 0.3012188971042633, 0.07776078581809998, 0.11717414855957031, 0.2512454390525818, 0.44061145186424255, 0.12461359798908234, 0.4159942865371704, 0.2479582130908966, 0.26588964462280273, 0.37404027581214905, 0.3919767439365387], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_30dd10d0c07f20d38e050d6f695094d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([4, 16, 240], dtype='float16', min=0, max=0.5),
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f29e5e959f516ec0ee28f7dbe3215b6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40192c1e65b36dea8f489ebf100b854
    def get_inputs(self):
        return [
            paddle.uniform([1, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_05039c3329edede44d1e8990518d8d56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 2304, 512], dtype='float16', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_debd4858042661a0c4b942216d741d31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3b63948a1844f93e58f6ca22a37c37a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a254031597d8288871b0422f0fa8e363
    def get_inputs(self):
        return [
            paddle.uniform([1, 3136, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9bb5eac8214120bb406299350520c7d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b845c90b460d5eeec7f3d2c4aea0318d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 2304, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_784b769e33775a1d4958f6d54695136d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 160], dtype='float16', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bfa81efa831236c1011a3fbaf7c7c608(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a254031597d8288871b0422f0fa8e363
    def get_inputs(self):
        return [
            paddle.uniform([4, 256, 144], dtype='float16', min=0, max=0.5),
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_30a04418c7a140355ef6be2df3090622(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8aff9b3db87ecc0f80859752c8d1af34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 768], dtype='float16', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c9e14311a3eb49e132917fc3e337ba31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2948295474052429, 0.25656041502952576, 0.02464757114648819, 0.476041316986084, 0.19090557098388672, 0.4085657298564911, 0.49969279766082764, 0.3506554961204529, 0.3696248233318329, 0.04948889836668968, 0.07433341443538666, 0.4906398057937622, 0.3546501100063324, 0.1728094220161438, 0.2735897898674011, 0.47032979130744934, 0.46880200505256653, 0.277831107378006, 0.3094508647918701, 0.3404964506626129, 0.06734950095415115, 0.022301096469163895, 0.235468327999115, 0.4539664387702942], dtype='float32').reshape([24]),
            paddle.to_tensor([0.22101391851902008, 0.07503221929073334, 0.398092120885849, 0.4442005753517151, 0.334077388048172, 0.14161737263202667, 0.29515066742897034, 0.15676969289779663, 0.07603178173303604, 0.2450062781572342, 0.06768611818552017, 0.04412431642413139, 0.15744595229625702, 0.19754409790039062, 0.35324999690055847, 0.015365750528872013, 0.10068827122449875, 0.38528066873550415, 0.4668484330177307, 0.22386804223060608, 0.3475293219089508, 0.23377925157546997, 0.037336625158786774, 0.3932895362377167], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8e3910b29aa529fd0121f7d524e57102(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1024], dtype='float16', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d64fe56cc7dd031f803d3aba96d9d8cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a254031597d8288871b0422f0fa8e363
    def get_inputs(self):
        return [
            paddle.uniform([1, 784, 128], dtype='float16', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1787fe966317706317d7c88386d3c421(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a254031597d8288871b0422f0fa8e363
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 256], dtype='float16', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_163bb28289c79e8c323140afe5001cf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a15a63ccd3ba6a7c452b22c08ee43e47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 256], dtype='float16', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9432716958c98ab24546417f96b1b852(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40192c1e65b36dea8f489ebf100b854
    def get_inputs(self):
        return [
            paddle.uniform([1, 3136, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_654ec34c00321709ead29b4d7cb4eab6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4075280427932739, 0.28762948513031006, 0.14922772347927094, 0.16289710998535156, 0.36465883255004883, 0.06658970564603806, 0.3865613043308258, 0.037496376782655716, 0.4020816385746002, 0.38956156373023987, 0.03481167182326317, 0.0018752855248749256, 0.07107016444206238, 0.13412366807460785, 0.34783926606178284, 0.2869226336479187, 0.3445194363594055, 0.27523764967918396, 0.3691956102848053, 0.23720872402191162, 0.10267768800258636, 0.16095612943172455, 0.0019124711398035288, 0.46512287855148315], dtype='float32').reshape([24]),
            paddle.to_tensor([0.08091019093990326, 0.04433393105864525, 0.058001041412353516, 0.001132540637627244, 0.2740916907787323, 0.38383543491363525, 0.08343417197465897, 0.13556107878684998, 0.018043357878923416, 0.49788594245910645, 0.38847774267196655, 0.24051205813884735, 0.09503314644098282, 0.29158028960227966, 0.4197055995464325, 0.30534827709198, 0.49423396587371826, 0.19452549517154694, 0.09706449508666992, 0.2676755487918854, 0.4134049713611603, 0.16105583310127258, 0.14482448995113373, 0.30570554733276367], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5f29ea2412fd0efda81a490273872661(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.11138676851987839, 0.47739094495773315, 0.1473296582698822, 0.23400244116783142, 0.2873220145702362, 0.3124854266643524, 0.40942201018333435, 0.34808632731437683, 0.04339507594704628, 0.26135051250457764, 0.05586106330156326, 0.4238629937171936, 0.4491181969642639, 0.4265049397945404, 0.4986898899078369, 0.417705237865448, 0.48625999689102173, 0.3012504577636719, 0.45965704321861267, 0.15417903661727905, 0.10371867567300797, 0.43387138843536377, 0.31557005643844604, 0.03589140251278877], dtype='float32').reshape([24]),
            paddle.to_tensor([0.42013609409332275, 0.42081624269485474, 0.15403145551681519, 0.16094908118247986, 0.3998282551765442, 0.02676352486014366, 0.0278155617415905, 0.4778689444065094, 0.449493408203125, 0.4048968255519867, 0.37809595465660095, 0.08318862318992615, 0.20823676884174347, 0.25754427909851074, 0.35324618220329285, 0.0029195775277912617, 0.29552334547042847, 0.4850471317768097, 0.4745393991470337, 0.35328713059425354, 0.011256619356572628, 0.4099480211734772, 0.3874702453613281, 0.31143122911453247], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0697324beff840eda046dc156c74faee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b5740e06b324122e623da5f0602df669(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 2048], dtype='float32', min=0, max=0.5),
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_77acfab37a0f0bf710b1eb68dce59f8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9bc3d61d0989da60b928e4edaf8c8a88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([4, 64, 192], dtype='float16', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_90254402f8d31f2bf1325fe100ba91b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 2304, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_78f1a68b48b46e5ac832f7b73de964df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2733425498008728, 0.2886544466018677, 0.12506115436553955, 0.2635963261127472, 0.40912050008773804, 0.14918892085552216, 0.36996743083000183, 0.40817272663116455, 0.3628562390804291, 0.022754129022359848, 0.03988342732191086, 0.30717965960502625, 0.47920846939086914, 0.183665931224823, 0.3315567076206207, 0.3608635663986206, 0.030695747584104538, 0.11753777414560318, 0.015341474674642086, 0.3680242598056793, 0.2977781295776367, 0.42913395166397095, 0.10832520574331284, 0.28080159425735474], dtype='float32').reshape([24]),
            paddle.to_tensor([0.3170471489429474, 0.22752729058265686, 0.3965907394886017, 0.47981470823287964, 0.34056150913238525, 0.45482271909713745, 0.10854624211788177, 0.44269371032714844, 0.0010614944621920586, 0.3491353690624237, 0.37615808844566345, 0.2246566265821457, 0.33170726895332336, 0.2919798493385315, 0.06577467173337936, 0.13128428161144257, 0.3963892459869385, 0.1373082846403122, 0.087900310754776, 0.3702240586280823, 0.029022011905908585, 0.16377879679203033, 0.4532797038555145, 0.25386375188827515], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c99f7c21e08108d598f742a4ae77976d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4942370355129242, 0.24279873073101044, 0.07200321555137634, 0.19086234271526337, 0.37410399317741394, 0.4693959653377533, 0.42202505469322205, 0.3212796449661255, 0.3400349020957947, 0.033894605934619904, 0.4438764750957489, 0.0531991608440876, 0.29941052198410034, 0.37350213527679443, 0.3689129054546356, 0.2065431773662567, 0.2327260971069336, 0.34288451075553894, 0.2392301708459854, 0.04949093237519264, 0.14563585817813873, 0.45946016907691956, 0.30910953879356384, 0.4555950462818146], dtype='float32').reshape([24]),
            paddle.to_tensor([0.4361044466495514, 0.2024260312318802, 0.04483854025602341, 0.23142854869365692, 0.17983077466487885, 0.21591337025165558, 0.45661216974258423, 0.12247885018587112, 0.005528866779059172, 0.01618219166994095, 0.3145701289176941, 0.21036936342716217, 0.2469642609357834, 0.39734336733818054, 0.14492493867874146, 0.15792416036128998, 0.29649606347084045, 0.24214282631874084, 0.051815465092659, 0.3278498947620392, 0.37583300471305847, 0.0743638202548027, 0.4275074899196625, 0.457813560962677], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c14954bd589a11e15bb22a55833316d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40192c1e65b36dea8f489ebf100b854
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6b6af2d30d053cdd4129fd3af99d6b5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 768], dtype='float16', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_45f042834f8efee9797be5b0c54afe9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a254031597d8288871b0422f0fa8e363
    def get_inputs(self):
        return [
            paddle.uniform([4, 64, 192], dtype='float16', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6a442445212ee5d8b8e912cadb9b421e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 128], dtype='float16', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2da5970fd032c8a9fb759947676d4f19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3136, 96], dtype='float16', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ef54c22074a7c6047ccad3a70b0a7563(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2749462425708771, 0.1498604416847229, 0.3775080144405365, 0.42425140738487244, 0.49972760677337646, 0.24153651297092438, 0.35319769382476807, 0.14112374186515808, 0.24404770135879517, 0.4066070020198822, 0.18401911854743958, 0.40190380811691284, 0.34167197346687317, 0.296133816242218, 0.44441258907318115, 0.4292530119419098, 0.34508177638053894, 0.4548507034778595, 0.4349629878997803, 0.15553048253059387, 0.48401716351509094, 0.2851331830024719, 0.48180171847343445, 0.19099731743335724], dtype='float32').reshape([24]),
            paddle.to_tensor([0.08021919429302216, 0.14500804245471954, 0.2378561794757843, 0.1521240621805191, 0.0067772469483315945, 0.41196808218955994, 0.26628732681274414, 0.18496370315551758, 0.040576208382844925, 0.33428266644477844, 0.3456243872642517, 0.3837692141532898, 0.2483023852109909, 0.0924777090549469, 0.18694263696670532, 0.0643448531627655, 0.2472136914730072, 0.1790827065706253, 0.3531460464000702, 0.2753889858722687, 0.24934150278568268, 0.3769308924674988, 0.12358508259057999, 0.1369844228029251], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f834e5811e46f5da82070c3c4f4a543e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7b298efaf36b77ac7da06fb582b1f908(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 3136, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5825e266e9cd9fec2310d23e44e197a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.11822105199098587, 0.35403379797935486, 0.09216026216745377, 0.4268139600753784, 0.3254181742668152, 0.10379862040281296, 0.2708767056465149, 0.45560401678085327, 0.1446813941001892, 0.025853773579001427, 0.4610397219657898, 0.10393553972244263, 0.25771042704582214, 0.34118756651878357, 0.43695932626724243, 0.2615312933921814, 0.4114696681499481, 0.20956392586231232, 0.4371609091758728, 0.23120185732841492, 0.4584100544452667, 0.06640075892210007, 0.46699827909469604, 0.26636576652526855], dtype='float32').reshape([24]),
            paddle.to_tensor([0.41902992129325867, 0.16590248048305511, 0.3327663242816925, 0.42059892416000366, 0.09136369079351425, 0.43001842498779297, 0.15384966135025024, 0.299831360578537, 0.17848137021064758, 0.22327029705047607, 0.19837695360183716, 0.034178365021944046, 0.1374148726463318, 0.12881284952163696, 0.13214822113513947, 0.4897172749042511, 0.1868322193622589, 0.48641303181648254, 0.07325220853090286, 0.1059931293129921, 0.18990863859653473, 0.10014212876558304, 0.26787856221199036, 0.3915049135684967], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7ddf319ba5c8823fbba947f0a7bf3827(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 784, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a1b773b88682ed798a19d42f3d23f275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.20196907222270966, 0.3077658414840698, 0.30164971947669983, 0.3887242376804352, 0.2066408097743988, 0.1929297298192978, 0.42979133129119873, 0.3063250780105591, 0.15751884877681732, 0.29275304079055786, 0.3532087802886963, 0.20203478634357452, 0.32315364480018616, 0.16049759089946747, 0.16638819873332977, 0.383476197719574, 0.49989399313926697, 0.4861551523208618, 0.3152659237384796, 0.08918523788452148, 0.2502291202545166, 0.4488310217857361, 0.26767122745513916, 0.21805009245872498], dtype='float32').reshape([24]),
            paddle.to_tensor([0.44470223784446716, 0.2137417495250702, 0.03797849640250206, 0.0681903213262558, 0.24024896323680878, 0.3687047064304352, 0.30915430188179016, 0.06465546786785126, 0.21601150929927826, 0.07825257629156113, 0.3345701992511749, 0.12046723067760468, 0.33100271224975586, 0.20086514949798584, 0.17381206154823303, 0.14239880442619324, 0.3918207585811615, 0.4170031249523163, 0.29186883568763733, 0.239900141954422, 0.33178573846817017, 0.23659555613994598, 0.2637886703014374, 0.33635568618774414], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_86b9d202767ab7b32ace8f1ac1404201(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.21603699028491974, 0.457147479057312, 0.32761427760124207, 0.43386441469192505, 0.3132554292678833, 0.12883122265338898, 0.34900692105293274, 0.15262621641159058, 0.14229939877986908, 0.0941244587302208, 0.13210459053516388, 0.07716771215200424, 0.17410129308700562, 0.42766106128692627, 0.28403162956237793, 0.45614659786224365, 0.022199802100658417, 0.2599000930786133, 0.11217594146728516, 0.3731764554977417, 0.2556445002555847, 0.1427105963230133, 0.28403982520103455, 0.18033753335475922], dtype='float32').reshape([24]),
            paddle.to_tensor([0.08212942630052567, 0.4710656702518463, 0.4386807978153229, 0.43541672825813293, 0.1459144949913025, 0.016623863950371742, 0.2603388726711273, 0.2934587001800537, 0.4624547064304352, 0.21479105949401855, 0.22284936904907227, 0.26773396134376526, 0.2879648208618164, 0.22528375685214996, 0.3968689441680908, 0.41876912117004395, 0.3529714345932007, 0.29852938652038574, 0.34880825877189636, 0.3127513527870178, 0.25624874234199524, 0.30797886848449707, 0.21493294835090637, 0.2662631869316101], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_71ad6673c740b6e41df1f6f6b9cf5335(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 2048], dtype='float16', min=0, max=0.5),
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b6e24ac6f5ba560174e5a9e21dd779c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 784, 128], dtype='float16', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7d798bd68ea3ce60c19c4ca5804a4f51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([4, 256, 144], dtype='float16', min=0, max=0.5),
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dba092ea447ecb281584a14038679d2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_837d44cb263ce58ba68ec1c4c10d1808(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.13104449212551117, 0.001003191340714693, 0.0004518513451330364, 0.3053096830844879, 0.4612765312194824, 0.2768879532814026, 0.09664745628833771, 0.22984181344509125, 0.3971695303916931, 0.26662760972976685, 0.03428388014435768, 0.2252470850944519, 0.16367743909358978, 0.22451382875442505, 0.4354277551174164, 0.18384778499603271, 0.2191624492406845, 0.34630006551742554, 0.1290626972913742, 0.03872497007250786, 0.03579448163509369, 0.37888261675834656, 0.09405921399593353, 0.19688816368579865], dtype='float32').reshape([24]),
            paddle.to_tensor([0.22849002480506897, 0.22476203739643097, 0.2837699353694916, 0.1893600970506668, 0.23464249074459076, 0.29270437359809875, 0.043193090707063675, 0.17402663826942444, 0.20343337953090668, 0.13933992385864258, 0.08156461268663406, 0.2924439609050751, 0.39190566539764404, 0.001491321949288249, 0.2799133360385895, 0.014528901316225529, 0.45894384384155273, 0.3823985159397125, 0.4985741972923279, 0.1520860642194748, 0.07000112533569336, 0.43293774127960205, 0.475347638130188, 0.21308377385139465], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_197513d5351c14c61c3d614aca696726(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.08189158141613007, 0.2640123665332794, 0.40644341707229614, 0.38590019941329956, 0.3514549434185028, 0.1318885236978531, 0.015498715452849865, 0.10658221691846848, 0.4806549549102783, 0.094437375664711, 0.07244237512350082, 0.4244176149368286, 0.1667337417602539, 0.30966898798942566, 0.33457323908805847, 0.1593007892370224, 0.2961292266845703, 0.05713701248168945, 0.43964460492134094, 0.12787842750549316, 0.15702253580093384, 0.022814583033323288, 0.18830177187919617, 0.2869359850883484], dtype='float32').reshape([24]),
            paddle.to_tensor([0.17696243524551392, 0.20154716074466705, 0.35040247440338135, 0.03410183638334274, 0.1347581446170807, 0.12588974833488464, 0.33610716462135315, 0.17126868665218353, 0.15495172142982483, 0.3760520815849304, 0.07632195204496384, 0.32297757267951965, 0.25233492255210876, 0.15374934673309326, 0.46340006589889526, 0.47967272996902466, 0.4464777410030365, 0.22481876611709595, 0.02110927365720272, 0.39305034279823303, 0.21820755302906036, 0.24352051317691803, 0.07854853570461273, 0.09641434997320175], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a252d558d05c006dc62b3eebbb72d1d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_51190a10959f94de2d7e5a01a683d141(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3032447397708893, 0.2310774326324463, 0.3592204749584198, 0.2898314595222473, 0.4603426456451416, 0.17551076412200928, 0.15908095240592957, 0.05398900806903839, 0.3695704936981201, 0.4119839370250702, 0.2586297392845154, 0.011514166370034218, 0.3243766725063324, 0.415735125541687, 0.4045689105987549, 0.1378740668296814, 0.12950938940048218, 0.06598066538572311, 0.0520794577896595, 0.4184545874595642, 0.019061435014009476, 0.15277910232543945, 0.46421897411346436, 0.07483349740505219], dtype='float32').reshape([24]),
            paddle.to_tensor([0.3027246296405792, 0.18151690065860748, 0.13708433508872986, 0.2975739538669586, 0.15942570567131042, 0.19100406765937805, 0.4496937394142151, 0.1999097466468811, 0.34864258766174316, 0.07857060432434082, 0.12883058190345764, 0.37036246061325073, 0.4196438193321228, 0.48509618639945984, 0.4795878231525421, 0.35608184337615967, 0.15370695292949677, 0.09583970904350281, 0.0893140584230423, 0.4857872724533081, 0.08927138894796371, 0.14895647764205933, 0.061342813074588776, 0.23118531703948975], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_aabf0a08eced233cd94e12d0d878c8d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 2304, 256], dtype='float16', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_46394158396be74a43b83cbb231a8780(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.003355184569954872, 0.08527450263500214, 0.49113503098487854, 0.06987988203763962, 0.3817385137081146, 0.057965606451034546, 0.28964295983314514, 0.37287047505378723, 0.20860333740711212, 0.060067348182201385, 0.2851361036300659, 0.12783615291118622, 0.07190290838479996, 0.40754643082618713, 0.36866819858551025, 0.20896939933300018, 0.21901348233222961, 0.16887909173965454, 0.2644297182559967, 0.303053081035614, 0.1525713950395584, 0.06794065237045288, 0.2204035073518753, 0.30499154329299927], dtype='float32').reshape([24]),
            paddle.to_tensor([0.4337674379348755, 0.12172119319438934, 0.33910325169563293, 0.2982933819293976, 0.3377218544483185, 0.10523975640535355, 0.29910847544670105, 0.2660064697265625, 0.1492518037557602, 0.004364924039691687, 0.43609419465065, 0.3189098536968231, 0.04427649453282356, 0.37041908502578735, 0.1509520411491394, 0.037587255239486694, 0.3917332589626312, 0.24273355305194855, 0.38657689094543457, 0.04726611077785492, 0.41798269748687744, 0.25149938464164734, 0.1313762068748474, 0.017108848318457603], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_32c80d45f30a19c3d9ce4785182e4c10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 160], dtype='float16', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7bd5cab5baee60f38ff5d0ca296b4098(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.22059546411037445, 0.08107399195432663, 0.3599371910095215, 0.06091917306184769, 0.13248299062252045, 0.48933812975883484, 0.35126540064811707, 0.40275704860687256, 0.3335936367511749, 0.4690301716327667, 0.10812464356422424, 0.21098940074443817, 0.3509853184223175, 0.08305541425943375, 0.29790058732032776, 0.28631922602653503, 0.41525518894195557, 0.19078561663627625, 0.30999377369880676, 0.17886129021644592, 0.3645116090774536, 0.30231353640556335, 0.27994534373283386, 0.4795728623867035], dtype='float32').reshape([24]),
            paddle.to_tensor([0.49116116762161255, 0.2589567303657532, 0.21399252116680145, 0.4345894157886505, 0.06589052826166153, 0.032483458518981934, 0.4032805562019348, 0.2811431586742401, 0.18462687730789185, 0.3623270094394684, 0.3536543548107147, 0.32368308305740356, 0.24250781536102295, 0.32686877250671387, 0.17247992753982544, 0.3505391776561737, 0.019868867471814156, 0.07206461578607559, 0.17273980379104614, 0.18356630206108093, 0.45990994572639465, 0.47260043025016785, 0.11956190317869186, 0.44359534978866577], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b7396b367a995cb31a2e44fe241f1b81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40192c1e65b36dea8f489ebf100b854
    def get_inputs(self):
        return [
            paddle.uniform([1, 3136, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_47d0666fd38957362ade6d3da00c758f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.07777924090623856, 0.1877835988998413, 0.44796645641326904, 0.13612505793571472, 0.44554564356803894, 0.19828172028064728, 0.4584275484085083, 0.48967087268829346, 0.27936676144599915, 0.12289503216743469, 0.17118236422538757, 0.02821151539683342, 0.32624655961990356, 0.22541551291942596, 0.47783684730529785, 0.3456783592700958, 0.18602311611175537, 0.06742773205041885, 0.18299618363380432, 0.48287785053253174, 0.3348250389099121, 0.44649025797843933, 0.36535608768463135, 0.3382311761379242], dtype='float32').reshape([24]),
            paddle.to_tensor([0.09664707630872726, 0.27151766419410706, 0.46327680349349976, 0.22908218204975128, 0.46835148334503174, 0.43873128294944763, 0.04229975864291191, 0.4763283431529999, 0.3355560004711151, 0.0445965901017189, 0.44537651538848877, 0.07344529777765274, 0.030793532729148865, 0.2414197474718094, 0.45863765478134155, 0.3339186906814575, 0.2636232376098633, 0.19587001204490662, 0.4803667366504669, 0.26928606629371643, 0.23129412531852722, 0.05639997124671936, 0.4081173837184906, 0.26974520087242126], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e003c55fba6b7ef2403e145f42433d3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40192c1e65b36dea8f489ebf100b854
    def get_inputs(self):
        return [
            paddle.uniform([1, 784, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_30274c3e2b559ced2aff2bef2ff94b9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4821987450122833, 0.265827476978302, 0.35885751247406006, 0.4651005268096924, 0.2871500849723816, 0.09241783618927002, 0.23046033084392548, 0.2547864615917206, 0.2563897967338562, 0.44208091497421265, 0.27567529678344727, 0.03303602337837219, 0.02879159152507782, 0.29305702447891235, 0.43727824091911316, 0.09752491116523743, 0.1959545612335205, 0.09608925879001617, 0.33840739727020264, 0.3394774794578552, 0.362621933221817, 0.361385315656662, 0.2533928155899048, 0.1054086685180664], dtype='float32').reshape([24]),
            paddle.to_tensor([0.2189566045999527, 0.1663825958967209, 0.3713485300540924, 0.4094517230987549, 0.1551073044538498, 0.41604241728782654, 0.13740676641464233, 0.006408956367522478, 0.3149452805519104, 0.2914600074291229, 0.303353488445282, 0.0042930361814796925, 0.21233555674552917, 0.2439674288034439, 0.3981761634349823, 0.31534624099731445, 0.17092184722423553, 0.111934132874012, 0.1160784587264061, 0.4051275849342346, 0.2443031668663025, 0.26640191674232483, 0.14770035445690155, 0.2985193431377411], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f4e4ad8b41f01844263095d42dd5df65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3569045960903168, 0.4219363033771515, 0.3463732600212097, 0.24545809626579285, 0.15596434473991394, 0.4199153184890747, 0.1694270521402359, 0.010177422314882278, 0.27669385075569153, 0.2522677481174469, 0.05690763145685196, 0.0931222066283226, 0.07311534881591797, 0.39175882935523987, 0.39445561170578003, 0.10012450814247131, 0.22885538637638092, 0.3355445861816406, 0.3250260055065155, 0.16773413121700287, 0.16229140758514404, 0.39997437596321106, 0.2962440550327301, 0.2199307531118393], dtype='float32').reshape([24]),
            paddle.to_tensor([0.19937656819820404, 0.2022133618593216, 0.4590785801410675, 0.07124201208353043, 0.2472957819700241, 0.253879189491272, 0.045171111822128296, 0.2538730204105377, 0.177532359957695, 0.31650349497795105, 0.004671516362577677, 0.23363719880580902, 0.17990192770957947, 0.21386675536632538, 0.31672754883766174, 0.442676305770874, 0.06987858563661575, 0.3340548872947693, 0.14056098461151123, 0.2355816662311554, 0.38369229435920715, 0.12451627105474472, 0.3545750081539154, 0.38873907923698425], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6f1f3c1f117e705810d31caa8fc875cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.35858413577079773, 0.34788861870765686, 0.2857751250267029, 0.25739121437072754, 0.49368444085121155, 0.3644040524959564, 0.1073170006275177, 0.18009315431118011, 0.19283881783485413, 0.4494086503982544, 0.1005583107471466, 0.445462703704834, 0.19405803084373474, 0.4874476492404938, 0.08407430350780487, 0.2414054274559021, 0.2729944884777069, 0.3681156635284424, 0.09988174587488174, 0.08254747092723846, 0.2911469638347626, 0.3688807189464569, 0.48850077390670776, 0.12959858775138855], dtype='float32').reshape([24]),
            paddle.to_tensor([0.289671927690506, 0.4417371451854706, 0.45394620299339294, 0.038958366960287094, 0.39196792244911194, 0.31908389925956726, 0.3375179171562195, 0.43845829367637634, 0.4918230473995209, 0.2424422949552536, 0.04904443025588989, 0.27253976464271545, 0.252619206905365, 0.23734422028064728, 0.23768794536590576, 0.3596724569797516, 0.3481784462928772, 0.30672597885131836, 0.18406398594379425, 0.09672915190458298, 0.3545641303062439, 0.23840880393981934, 0.08152862638235092, 0.19002404808998108], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9e6130b20f139429ad2c46ac349f4a45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3136, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_57599fae126ba5e4abbb4a4da8d53a0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.18366307020187378, 0.12426714599132538, 0.4972164034843445, 0.062458768486976624, 0.4127230644226074, 0.20901690423488617, 0.0540064238011837, 0.4543137550354004, 0.41377779841423035, 0.13937672972679138, 0.2189370095729828, 0.2536654770374298, 0.04694389924407005, 0.25529488921165466, 0.052069131284952164, 0.3095897436141968, 0.3341934382915497, 0.23582299053668976, 0.1625525802373886, 0.1395953744649887, 0.4057944118976593, 0.41055068373680115, 0.35570409893989563, 0.35884156823158264], dtype='float32').reshape([24]),
            paddle.to_tensor([0.02742994949221611, 0.31682878732681274, 0.49627548456192017, 0.06683100759983063, 0.15778422355651855, 0.04458969458937645, 0.49738025665283203, 0.4496666491031647, 0.4337845742702484, 0.1711878627538681, 0.2398301362991333, 0.042817216366529465, 0.37853601574897766, 0.45221853256225586, 0.026275746524333954, 0.0619528666138649, 0.4249951243400574, 0.06234719231724739, 0.031480420380830765, 0.24545733630657196, 0.46106672286987305, 0.4392014443874359, 0.20094449818134308, 0.3954869210720062], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f69470cae9a7079f2d79da3a3878b85b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.29579704999923706, 0.3934045135974884, 0.2071964293718338, 0.03914392367005348, 0.3086318373680115, 0.2769215404987335, 0.10099522024393082, 0.005522829480469227, 0.3575942814350128, 0.1584794819355011, 0.3809984624385834, 0.471023291349411, 0.37874114513397217, 0.20067955553531647, 0.11812857538461685, 0.46527349948883057, 0.08282085508108139, 0.3970787227153778, 0.36169201135635376, 0.24797359108924866, 0.47906413674354553, 0.2564951479434967, 0.20490019023418427, 0.09064430743455887], dtype='float32').reshape([24]),
            paddle.to_tensor([0.11117620766162872, 0.10596942901611328, 0.20791016519069672, 0.038244303315877914, 0.4680157005786896, 0.3131575584411621, 0.2714845836162567, 0.10299527645111084, 0.48713934421539307, 0.4871602952480316, 0.28845202922821045, 0.25746169686317444, 0.14901545643806458, 0.19631822407245636, 0.2599518597126007, 0.44552749395370483, 0.4309538006782532, 0.19098258018493652, 0.3939175605773926, 0.4217641055583954, 0.45723047852516174, 0.10886441916227341, 0.23225022852420807, 0.22712258994579315], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fba0b45bc266d76ef6dd4bb87d1f118e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.014243364334106445, 0.2794649004936218, 0.32717716693878174, 0.47490912675857544, 0.22570741176605225, 0.16042979061603546, 0.37896236777305603, 0.030417995527386665, 0.4506978988647461, 0.11219741404056549, 0.20871514081954956, 0.32348334789276123, 0.15902097523212433, 0.04215121269226074, 0.33045271039009094, 0.0210556723177433, 0.49260541796684265, 0.34995031356811523, 0.09651444107294083, 0.13955333828926086, 0.02192113548517227, 0.16689813137054443, 0.48275765776634216, 0.11836960166692734], dtype='float32').reshape([24]),
            paddle.to_tensor([0.0507824569940567, 0.486506849527359, 0.46001583337783813, 0.1119002103805542, 0.2957666516304016, 0.34965643286705017, 0.33913567662239075, 0.21292796730995178, 0.40829792618751526, 0.2172970473766327, 0.3668845295906067, 0.07309071719646454, 0.13398326933383942, 0.02183481864631176, 0.0208249744027853, 0.0062780617736279964, 0.22229428589344025, 0.10834936797618866, 0.41508275270462036, 0.056368354707956314, 0.03811442479491234, 0.001178551698103547, 0.19034916162490845, 0.04245162382721901], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_be1eebe0a5cf5e0352cbaef41d035d88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40192c1e65b36dea8f489ebf100b854
    def get_inputs(self):
        return [
            paddle.uniform([4, 64, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_60d1bf28d4b4b8559d536999633f24cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.16475403308868408, 0.4828357398509979, 0.4258727729320526, 0.22399568557739258, 0.14013613760471344, 0.26562273502349854, 0.4289284348487854, 0.4146000146865845, 0.1591757982969284, 0.36893364787101746, 0.18424677848815918, 0.40686145424842834, 0.2382490634918213, 0.20175544917583466, 0.12711192667484283, 0.389931857585907, 0.06601797044277191, 0.42960959672927856, 0.3454381823539734, 0.0585675872862339, 0.44052815437316895, 0.37792930006980896, 0.1036595031619072, 0.27953049540519714], dtype='float32').reshape([24]),
            paddle.to_tensor([0.05428473278880119, 0.42193880677223206, 0.04669639840722084, 0.2255106270313263, 0.1590641438961029, 0.42216768860816956, 0.43623289465904236, 0.006709329783916473, 0.009879864752292633, 0.4408259391784668, 0.03323203697800636, 0.050958674401044846, 0.18800625205039978, 0.43866127729415894, 0.126575767993927, 0.44613292813301086, 0.4748630225658417, 0.2567873001098633, 0.21956348419189453, 0.46179062128067017, 0.13494503498077393, 0.11803732067346573, 0.0688214972615242, 0.06921400874853134], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bc0b03dc608813709d94c4d0cd72cfa1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.25710543990135193, 0.3526293933391571, 0.4140579402446747, 0.4722927212715149, 0.2351761758327484, 0.37296032905578613, 0.0974608063697815, 0.19325564801692963, 0.011694647371768951, 0.16122131049633026, 0.04489336907863617, 0.1953994631767273, 0.23894086480140686, 0.1467740833759308, 0.15643905103206635, 0.010504195466637611, 0.17769835889339447, 0.0664338693022728, 0.36663544178009033, 0.03689739108085632, 0.3257657587528229, 0.22407668828964233, 0.41362735629081726, 0.22624360024929047], dtype='float32').reshape([24]),
            paddle.to_tensor([0.3817916512489319, 0.3329784572124481, 0.4374532997608185, 0.4319269061088562, 0.19798001646995544, 0.10686233639717102, 0.16284531354904175, 0.2656308710575104, 0.1156698539853096, 0.22367294132709503, 0.3040522634983063, 0.48494473099708557, 0.48920366168022156, 0.33050382137298584, 0.11264272034168243, 0.23953868448734283, 0.4199221432209015, 0.42665019631385803, 0.21511241793632507, 0.4858913719654083, 0.030990809202194214, 0.07033313065767288, 0.07851997762918472, 0.18951284885406494], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ebaa666f4db551db1ebb5a30f77d4b2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.13936637341976166, 0.44048917293548584, 0.34544068574905396, 0.226340189576149, 0.11526307463645935, 0.30719825625419617, 0.2041853368282318, 0.290505588054657, 0.2068309783935547, 0.1575533002614975, 0.37899428606033325, 0.40094995498657227, 0.1259523332118988, 0.37002459168434143, 0.3306317925453186, 0.22001951932907104, 0.013940393924713135, 0.3108084499835968, 0.2097664624452591, 0.10541463643312454, 0.3320286273956299, 0.1915835440158844, 0.1417192965745926, 0.06755843013525009], dtype='float32').reshape([24]),
            paddle.to_tensor([0.1867671012878418, 0.04889613389968872, 0.47734805941581726, 0.1333237588405609, 0.07605227828025818, 0.4606616199016571, 0.4870370626449585, 0.06249965727329254, 0.4981561303138733, 0.4332980513572693, 0.11396840959787369, 0.34568482637405396, 0.15402203798294067, 0.2825644612312317, 0.3632968068122864, 0.36311984062194824, 0.13750992715358734, 0.2815728187561035, 0.39703822135925293, 0.08063396066427231, 0.13278478384017944, 0.11740539222955704, 0.3181067705154419, 0.1884966939687729], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_db1a7a1149486d1d2677f4fed6458810(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a254031597d8288871b0422f0fa8e363
    def get_inputs(self):
        return [
            paddle.uniform([4, 16, 240], dtype='float16', min=0, max=0.5),
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_766d793e168a77024acb958b967447e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1938668042421341, 0.07876656949520111, 0.09433778375387192, 0.4779868423938751, 0.34814438223838806, 0.4469517469406128, 0.43437811732292175, 0.159662127494812, 0.18601472675800323, 0.10298773646354675, 0.4830115735530853, 0.13508522510528564, 0.068057581782341, 0.23629240691661835, 0.0015584391076117754, 0.13886502385139465, 0.16179491579532623, 0.3291814625263214, 0.2886349856853485, 0.02491985447704792, 0.28625237941741943, 0.24895773828029633, 0.17609760165214539, 0.06893787533044815], dtype='float32').reshape([24]),
            paddle.to_tensor([0.049390219151973724, 0.2170150727033615, 0.2732374966144562, 0.4153362810611725, 0.06235528737306595, 0.20021355152130127, 0.18336232006549835, 0.06839175522327423, 0.2917318642139435, 0.3795059621334076, 0.10806344449520111, 0.16893772780895233, 0.03804459422826767, 0.05009950324892998, 0.2471439093351364, 0.4515213072299957, 0.03288698568940163, 0.4547441005706787, 0.23313893377780914, 0.12240840494632721, 0.22567322850227356, 0.03833785653114319, 0.37192702293395996, 0.20805618166923523], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c21559071a9b79f6e2482ff903ff5ed1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40192c1e65b36dea8f489ebf100b854
    def get_inputs(self):
        return [
            paddle.uniform([4, 16, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a97f53ece7862a0d3d8b88ba2e6021f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.18731826543807983, 0.4828099310398102, 0.25584864616394043, 0.01906595006585121, 0.20454615354537964, 0.07630125433206558, 0.004274432547390461, 0.4532151222229004, 0.19866101443767548, 0.20867544412612915, 0.12019315361976624, 0.22176975011825562, 0.2413586527109146, 0.07520231604576111, 0.35733622312545776, 0.379110723733902, 0.41227513551712036, 0.1334734708070755, 0.36011257767677307, 0.39803704619407654, 0.10016573220491409, 0.08654404431581497, 0.06558667123317719, 0.20992836356163025], dtype='float32').reshape([24]),
            paddle.to_tensor([0.3008992075920105, 0.06400731950998306, 0.16011938452720642, 0.2368287593126297, 0.48954954743385315, 0.07106383889913559, 0.018838316202163696, 0.3443790674209595, 0.4682483673095703, 0.12135474383831024, 0.06300223618745804, 0.4523089826107025, 0.2626247704029083, 0.04270777851343155, 0.13154040277004242, 0.010985239408910275, 0.31751522421836853, 0.2806195914745331, 0.4354870021343231, 0.45831331610679626, 0.1530156433582306, 0.3335305154323578, 0.10656460374593735, 0.18355105817317963], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_636a8ffa9baf161b787bb8c882c4ddf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b75e3738c82bb5489706f7e28561ff18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 32], dtype='float16', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3db2fb91b5ad696f52832fa063994055(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.010870775207877159, 0.03675542026758194, 0.24669773876667023, 0.05390876159071922, 0.316701203584671, 0.17424993216991425, 0.03908360376954079, 0.43361032009124756, 0.4712028503417969, 0.3705647587776184, 0.07935892790555954, 0.0350777767598629, 0.41310372948646545, 0.3556474447250366, 0.02611469477415085, 0.11116866767406464, 0.07265156507492065, 0.3031384348869324, 0.291408509016037, 0.49377259612083435, 0.17582684755325317, 0.07363982498645782, 0.09489616751670837, 0.35881510376930237], dtype='float32').reshape([24]),
            paddle.to_tensor([0.22165365517139435, 0.22210361063480377, 0.25007158517837524, 0.04683087021112442, 0.0700167864561081, 0.27537307143211365, 0.4492897391319275, 0.4553984999656677, 0.17673273384571075, 0.4537613093852997, 0.47669342160224915, 0.448720246553421, 0.4232642650604248, 0.0006090207607485354, 0.2675057649612427, 0.46675780415534973, 0.38979390263557434, 0.3350133001804352, 0.22259560227394104, 0.003308809595182538, 0.03921572491526604, 0.4619683623313904, 0.48399123549461365, 0.4778873026371002], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_31cefdebde78080edfe4a6256b3e4fc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_18cb7f96be46c4ada6233283b85a6cdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a254031597d8288871b0422f0fa8e363
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_279b55f08da06e40f3f76f1c9d3a1b89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.33882394433021545, 0.3488764464855194, 0.03562818467617035, 0.3791046738624573, 0.3421655297279358, 0.13304094970226288, 0.18232062458992004, 0.14956778287887573, 0.052485957741737366, 0.3806203305721283, 0.2293543964624405, 0.18278725445270538, 0.20754069089889526, 0.2983909249305725, 0.48287203907966614, 0.24650907516479492, 0.19480866193771362, 0.26233288645744324, 0.4814402759075165, 0.12110041826963425, 0.0721556767821312, 0.35488057136535645, 0.4377501904964447, 0.05875638872385025], dtype='float32').reshape([24]),
            paddle.to_tensor([0.3843730688095093, 0.1805105358362198, 0.40081948041915894, 0.1916782408952713, 0.28682342171669006, 0.44120410084724426, 0.417477011680603, 0.20276883244514465, 0.35341861844062805, 0.3918842673301697, 0.19456268846988678, 0.15923380851745605, 0.06132087856531143, 0.30830180644989014, 0.1771114468574524, 0.2213728427886963, 0.3380354642868042, 0.01230168528854847, 0.45821240544319153, 0.22713106870651245, 0.3093269169330597, 0.3850228786468506, 0.11108630895614624, 0.022809235379099846], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_422c7f9bfc21d18e9e509b18f74613e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 192], dtype='float16', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ec82031fbde097bb06c6c795cdbccb0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.07532916218042374, 0.1972924917936325, 0.4818250238895416, 0.34716910123825073, 0.1353168487548828, 0.12077910453081131, 0.4096485674381256, 0.42836514115333557, 0.14646293222904205, 0.21278241276741028, 0.46058011054992676, 0.2457815706729889, 0.3760378956794739, 0.046136524528265, 0.22805488109588623, 0.1155649796128273, 0.32413366436958313, 0.49832016229629517, 0.42351728677749634, 0.08922522515058517, 0.23585352301597595, 0.08159761875867844, 0.33681657910346985, 0.3886728286743164], dtype='float32').reshape([24]),
            paddle.to_tensor([0.186040461063385, 0.3528192639350891, 0.09311974048614502, 0.22022080421447754, 0.03183605521917343, 0.23108990490436554, 0.05479612946510315, 0.0018270742148160934, 0.2949790060520172, 0.3881951570510864, 0.11235623061656952, 0.3297078013420105, 0.4999403953552246, 0.18454866111278534, 0.4581121802330017, 0.43436485528945923, 0.07245106250047684, 0.09357249736785889, 0.30169519782066345, 0.2772696912288666, 0.3879294991493225, 0.00434198509901762, 0.10458900034427643, 0.4332750141620636], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5732c600c31f84ea985ab612a60f8310(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.15723566710948944, 0.2434181272983551, 0.012888051569461823, 0.0034653411712497473, 0.4338573217391968, 0.10937909781932831, 0.46160873770713806, 0.22581854462623596, 0.32471129298210144, 0.40940603613853455, 0.3611995279788971, 0.073187455534935, 0.025003740563988686, 0.385524719953537, 0.09263664484024048, 0.03777698799967766, 0.15941482782363892, 0.017981860786676407, 0.28543469309806824, 0.27014341950416565, 0.3705376386642456, 0.01776932366192341, 0.26192259788513184, 0.22764934599399567], dtype='float32').reshape([24]),
            paddle.to_tensor([0.37588319182395935, 0.4547136425971985, 0.1828862726688385, 0.1647234857082367, 0.04327135160565376, 0.018667694181203842, 0.2758021652698517, 0.4080827832221985, 0.10054291039705276, 0.4233485758304596, 0.3048818111419678, 0.1888388842344284, 0.45933932065963745, 0.35873204469680786, 0.45759496092796326, 0.45182090997695923, 0.17913800477981567, 0.12784822285175323, 0.2736762464046478, 0.4633031189441681, 0.150021493434906, 0.3244920074939728, 0.3048169016838074, 0.34645697474479675], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f2d4c4d5acd1e7ac74b3d39ca6cc9194(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.29522278904914856, 0.0950869619846344, 0.05496954545378685, 0.46835246682167053, 0.23117777705192566, 0.4735710918903351, 0.4322499930858612, 0.3404623568058014, 0.3642272353172302, 0.04208987578749657, 0.12262923270463943, 0.078940249979496, 0.1275501549243927, 0.4027003347873688, 0.0009189197444356978, 0.4557492434978485, 0.20845770835876465, 0.238201305270195, 0.0010032330173999071, 0.19651584327220917, 0.15796522796154022, 0.3953298032283783, 0.006185709033161402, 0.13807445764541626], dtype='float32').reshape([24]),
            paddle.to_tensor([0.38487422466278076, 0.030802318826317787, 0.46444231271743774, 0.22993767261505127, 0.15590202808380127, 0.07117673009634018, 0.0986221432685852, 0.3898000121116638, 0.21487857401371002, 0.10428028553724289, 0.34426283836364746, 0.1759105771780014, 0.0740404799580574, 0.4490010142326355, 0.13167668879032135, 0.06733668595552444, 0.14908458292484283, 0.058156583458185196, 0.330643892288208, 0.4362650513648987, 0.23603256046772003, 0.22355514764785767, 0.37005114555358887, 0.3322056829929352], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eeacf2af6ccd79983a9aae3af6a3b72d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3136, 32], dtype='float16', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0528608e2e10cf3357fc4ea2e22927df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1682705283164978, 0.3565743863582611, 0.1253589242696762, 0.16418196260929108, 0.3286559581756592, 0.48541876673698425, 0.2749499976634979, 0.4234136939048767, 0.08529394119977951, 0.13429801166057587, 0.07868754863739014, 0.00985343474894762, 0.06885045766830444, 0.35013991594314575, 0.22062887251377106, 0.14742974936962128, 0.39567428827285767, 0.3646829426288605, 0.24926216900348663, 0.2954123318195343, 0.01851140707731247, 0.020235564559698105, 0.179886594414711, 0.06986070424318314], dtype='float32').reshape([24]),
            paddle.to_tensor([0.22930927574634552, 0.36494237184524536, 0.18423578143119812, 0.46554070711135864, 0.1347815990447998, 0.20376929640769958, 0.25552263855934143, 0.2063751220703125, 0.40140387415885925, 0.4490581154823303, 0.39951521158218384, 0.0205830130726099, 0.2740081250667572, 0.48853519558906555, 0.2794617712497711, 0.15489323437213898, 0.44764864444732666, 0.02543565444648266, 0.002081912476569414, 0.2824811637401581, 0.1383700966835022, 0.12855780124664307, 0.056254513561725616, 0.35080331563949585], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a49b433d13ab039c288533b344b91068(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1726970374584198, 0.16181598603725433, 0.10104984790086746, 0.3545682728290558, 0.2902470827102661, 0.23746728897094727, 0.30638277530670166, 0.029487179592251778, 0.07071344554424286, 0.489443302154541, 0.4145127832889557, 0.13114866614341736, 0.007585819344967604, 0.006573534104973078, 0.29442575573921204, 0.21814538538455963, 0.0454707071185112, 0.459863543510437, 0.16221505403518677, 0.4264506697654724, 0.08130539208650589, 0.27876222133636475, 0.22132013738155365, 0.3513026833534241], dtype='float32').reshape([24]),
            paddle.to_tensor([0.29074880480766296, 0.24003681540489197, 0.0007325326441787183, 0.4447775185108185, 0.23070864379405975, 0.10364915430545807, 0.4267929196357727, 0.38438647985458374, 0.15183788537979126, 0.41052234172821045, 0.3081268072128296, 0.07744412869215012, 0.09051049500703812, 0.031048299744725227, 0.34466007351875305, 0.08504622429609299, 0.005109978839755058, 0.4140075445175171, 0.22425530850887299, 0.1307162195444107, 0.24897487461566925, 0.1401134878396988, 0.25636357069015503, 0.017722351476550102], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a0b81a7f2775a2a5ad300106e30c00cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.33938372135162354, 0.40610218048095703, 0.30375155806541443, 0.2319847196340561, 0.11537652462720871, 0.432975709438324, 0.22722597420215607, 0.2257462590932846, 0.41630417108535767, 0.443502277135849, 0.37145131826400757, 0.31059888005256653, 0.42944005131721497, 0.2317645400762558, 0.3220788240432739, 0.3324247896671295, 0.32443344593048096, 0.2245652675628662, 0.07454076409339905, 0.047343313694000244, 0.22435623407363892, 0.06307432055473328, 0.07455383241176605, 0.24440547823905945], dtype='float32').reshape([24]),
            paddle.to_tensor([0.2475200593471527, 0.3294788599014282, 0.07380232214927673, 0.4451567232608795, 0.14242061972618103, 0.27744394540786743, 0.4061700403690338, 0.257845401763916, 0.05833157151937485, 0.21847888827323914, 0.46438068151474, 0.16107966005802155, 0.08478806167840958, 0.3496074974536896, 0.004090358968824148, 0.3098233640193939, 0.35273170471191406, 0.3347354829311371, 0.19040662050247192, 0.3860618472099304, 0.2835719883441925, 0.2049616575241089, 0.437127947807312, 0.11209271848201752], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c21a28954184ea269537b498682e3448(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4525972604751587, 0.10570936650037766, 0.19949999451637268, 0.4838881194591522, 0.2377445548772812, 0.17726634442806244, 0.24631813168525696, 0.30529892444610596, 0.24530333280563354, 0.2580033838748932, 0.2084118276834488, 0.4892955720424652, 0.04473545029759407, 0.27066031098365784, 0.4945635199546814, 0.4525549113750458, 0.18380023539066315, 0.14176037907600403, 0.21209178864955902, 0.1922387033700943, 0.3587110936641693, 0.009046556428074837, 0.21292348206043243, 0.06784713268280029], dtype='float32').reshape([24]),
            paddle.to_tensor([0.036773622035980225, 0.12827908992767334, 0.4302771985530853, 0.35752516984939575, 0.2255377471446991, 0.3609851002693176, 0.37519529461860657, 0.17342035472393036, 0.2122988998889923, 0.34364548325538635, 0.41927674412727356, 0.038604021072387695, 0.42805013060569763, 0.39877498149871826, 0.08990813791751862, 0.38422563672065735, 0.08769549429416656, 0.4307102859020233, 0.08969058096408844, 0.3177984952926636, 0.31290921568870544, 0.28953176736831665, 0.12662701308727264, 0.30445700883865356], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_43d56a7955b45052270dc3f5b5fe5fff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.18257200717926025, 0.43181341886520386, 0.30533620715141296, 0.14209537208080292, 0.29281044006347656, 0.040524162352085114, 0.339503675699234, 0.12490895390510559, 0.002219052752479911, 0.24146921932697296, 0.1458093523979187, 0.4019382894039154, 0.26807624101638794, 0.3998703956604004, 0.350229948759079, 0.4934419095516205, 0.08212210237979889, 0.35577523708343506, 0.23343515396118164, 0.2782115340232849, 0.36634939908981323, 0.07314932346343994, 0.053259674459695816, 0.32733631134033203], dtype='float32').reshape([24]),
            paddle.to_tensor([0.13197894394397736, 0.13877402245998383, 0.18933744728565216, 0.4786091148853302, 0.24661923944950104, 0.021998360753059387, 0.4621764123439789, 0.15356013178825378, 0.19081975519657135, 0.4565843641757965, 0.4964907467365265, 0.45384037494659424, 0.14349815249443054, 0.368434876203537, 0.13210786879062653, 0.4741015136241913, 0.38814622163772583, 0.09981435537338257, 0.08217289298772812, 0.05007820948958397, 0.0030590062960982323, 0.04989596828818321, 0.3537556529045105, 0.37192681431770325], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_939ea0b6e80bd0bf42deec2698f19722(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40192c1e65b36dea8f489ebf100b854
    def get_inputs(self):
        return [
            paddle.uniform([4, 256, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6738a56407da6c57eb58334f306246be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3968702554702759, 0.4320816397666931, 0.3889198303222656, 0.30037838220596313, 0.11799021810293198, 0.4648837745189667, 0.03690313920378685, 0.05396295338869095, 0.46167248487472534, 0.28702905774116516, 0.4455452859401703, 0.46462130546569824, 0.46802258491516113, 0.1937229484319687, 0.3695943355560303, 0.0727655217051506, 0.0497843399643898, 0.3123270273208618, 0.15348221361637115, 0.05088169872760773, 0.30928343534469604, 0.14680342376232147, 0.48877212405204773, 0.037296999245882034], dtype='float32').reshape([24]),
            paddle.to_tensor([0.2778010964393616, 0.27052295207977295, 0.19602876901626587, 0.1684628129005432, 0.4654140770435333, 0.3379822075366974, 0.2950296401977539, 0.17189368605613708, 0.005182778928428888, 0.18551237881183624, 0.05109640210866928, 0.4282330274581909, 0.21332058310508728, 0.14822576940059662, 0.0646529421210289, 0.3894656300544739, 0.313083678483963, 0.03633873537182808, 0.410410612821579, 0.1478414684534073, 0.3513844311237335, 0.06024102866649628, 0.3374421000480652, 0.1095222607254982], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b2aba7f8396770f875433fb0d9a3866a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.30301332473754883, 0.4493446946144104, 0.4292582869529724, 0.3049544095993042, 0.08340997993946075, 0.3223626911640167, 0.1273801475763321, 0.17693401873111725, 0.04197666049003601, 0.49660390615463257, 0.4233519732952118, 0.37309715151786804, 0.31765690445899963, 0.14948458969593048, 0.004608043469488621, 0.2522963285446167, 0.21284528076648712, 0.05863044038414955, 0.07177890837192535, 0.32406264543533325, 0.2306806594133377, 0.10358590632677078, 0.1974070966243744, 0.32947397232055664], dtype='float32').reshape([24]),
            paddle.to_tensor([0.18616163730621338, 0.28802230954170227, 0.21353426575660706, 0.3143126964569092, 0.4350679814815521, 0.0990566536784172, 0.35628223419189453, 0.0036044223234057426, 0.12603428959846497, 0.2742798924446106, 0.49496158957481384, 0.23257498443126678, 0.44189727306365967, 0.305686891078949, 0.4327239990234375, 0.054918453097343445, 0.006196706555783749, 0.253266304731369, 0.32935842871665955, 0.025479452684521675, 0.24025894701480865, 0.3470144271850586, 0.33367231488227844, 0.440521240234375], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d6552586b651882f89033cf9f2fc8cdd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fc361031adc0ab7e7e4a62b958dd1dd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1024], dtype='float16', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c56c9b412cf8d9593cd06a2720284011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 96], dtype='float16', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cdbe80ba53f49ece406e76f9f67924d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.06041004881262779, 0.3167206943035126, 0.05374542996287346, 0.36268866062164307, 0.19197198748588562, 0.4946305751800537, 0.12886005640029907, 0.14391383528709412, 0.16145367920398712, 0.3001125752925873, 0.18410956859588623, 0.18207840621471405, 0.4029044806957245, 0.06203930452466011, 0.144857719540596, 0.2986207604408264, 0.2609812319278717, 0.32699164748191833, 0.18422389030456543, 0.48522546887397766, 0.15709753334522247, 0.37659984827041626, 0.18521958589553833, 0.14869895577430725], dtype='float32').reshape([24]),
            paddle.to_tensor([0.31494954228401184, 0.17287783324718475, 0.16401103138923645, 0.4613477289676666, 0.04905761033296585, 0.464922696352005, 0.11221131682395935, 0.2821165919303894, 0.29332780838012695, 0.019930627197027206, 0.2986953556537628, 0.35327333211898804, 0.3241746723651886, 0.2458716630935669, 0.4133024513721466, 0.07294643670320511, 0.20899417996406555, 0.20976218581199646, 0.1856888234615326, 0.425304114818573, 0.1442001610994339, 0.18219241499900818, 0.18387548625469208, 0.4614901840686798], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ad7caf926671775ce770b3df2d0d71f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.33545830845832825, 0.37854981422424316, 0.1908382773399353, 0.05427590012550354, 0.1394834965467453, 0.14096412062644958, 0.13863807916641235, 0.39346635341644287, 0.19579078257083893, 0.010661977343261242, 0.4447462558746338, 0.3711831569671631, 0.07698850333690643, 0.09032592922449112, 0.38182857632637024, 0.35339832305908203, 0.15278765559196472, 0.18944057822227478, 0.2009609043598175, 0.08842156827449799, 0.1725178062915802, 0.3571804165840149, 0.030272960662841797, 0.48736661672592163], dtype='float32').reshape([24]),
            paddle.to_tensor([0.285391241312027, 0.12025842815637589, 0.08677830547094345, 0.04854026809334755, 0.026385126635432243, 0.4224960207939148, 0.32764673233032227, 0.4977532923221588, 0.4031577408313751, 0.37871429324150085, 0.05479966849088669, 0.1505054086446762, 0.003360736183822155, 0.11643196642398834, 0.28942087292671204, 0.10076363384723663, 0.21567189693450928, 0.1498848795890808, 0.15887169539928436, 0.0488554947078228, 0.24201691150665283, 0.34989067912101746, 0.20457899570465088, 0.49740856885910034], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_24fc18f2d05b090df58f251d42345b99(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 320], dtype='float16'),
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            paddle.static.InputSpec(shape=[320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_661d5c734b828c9a043de7321f6d1c04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24fc18f2d05b090df58f251d42345b99
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 320], dtype='float16', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_bdec3fbaf88ffee720f6c968e9f14283(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 576, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a95186096a2daf3591659df92cd3633d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bdec3fbaf88ffee720f6c968e9f14283
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 24], dtype='float32'),
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            paddle.static.InputSpec(shape=[24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4888976c987e264c14ba48f588970713(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.007542538922280073, 0.2589527666568756, 0.487557977437973, 0.29581987857818604, 0.39868220686912537, 0.12857985496520996, 0.17470674216747284, 0.21117405593395233, 0.13679154217243195, 0.1589573174715042, 0.3821004033088684, 0.4808706045150757, 0.004323906730860472, 0.26371440291404724, 0.45760050415992737, 0.38391822576522827, 0.18062378466129303, 0.13580891489982605, 0.06980408728122711, 0.3521064817905426, 0.463852196931839, 0.033546581864356995, 0.12499123066663742, 0.3425736427307129], dtype='float32').reshape([24]),
            paddle.to_tensor([0.1796165406703949, 0.04581243172287941, 0.2864757776260376, 0.11248375475406647, 0.2057366818189621, 0.37921521067619324, 0.43359479308128357, 0.18197079002857208, 0.008127328008413315, 0.20194712281227112, 0.23519830405712128, 0.047192152589559555, 0.34459710121154785, 0.33789950609207153, 0.33767247200012207, 0.4074074923992157, 0.47153669595718384, 0.04866562783718109, 0.4100349545478821, 0.3305538296699524, 0.15910589694976807, 0.4810023009777069, 0.43099188804626465, 0.06479601562023163], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_c1810c147e9790f0c38d76040d9a876f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 197, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0558f21e8763d70103846b66b4c48263(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1810c147e9790f0c38d76040d9a876f
    def get_inputs(self):
        return [
            paddle.uniform([1, 197, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_f5f81246e867c3e233889d0268bf2278(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 196, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6d9077eb01ab74a54b64654816f50a1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5f81246e867c3e233889d0268bf2278
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_c9f087c5bb03024d10c0dd030843b854(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 576, 512], dtype='float16'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d6a3f08ba5637fdc2fa751974deb7d4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9f087c5bb03024d10c0dd030843b854
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 512], dtype='float16', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_3f2cfce654eb8814c3e8361e62d73ceb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 196, 320], dtype='float16'),
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            paddle.static.InputSpec(shape=[320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e0a0ce9d9a04d732e4b632379b57a6a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f2cfce654eb8814c3e8361e62d73ceb
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 320], dtype='float16', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_937f99ba5d70cd6325f7d53e0a367306(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_68982d1dabf2b78308598f8f129d610e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_937f99ba5d70cd6325f7d53e0a367306
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_68a63a83e8dd554783b3c9361f9867a1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 196, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8b6665dd964fc9364809c1619aabf914(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_68a63a83e8dd554783b3c9361f9867a1
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_f180a2767fb438719a209475fc419192(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 197, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_aaf9c901f800ebabed54730feaa5dd04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f180a2767fb438719a209475fc419192
    def get_inputs(self):
        return [
            paddle.uniform([1, 197, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_38658ec03168625761e5584f2e9d7d22(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 196, 384], dtype='float16'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_53f3c40c1423714b0618e7afe76964aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38658ec03168625761e5584f2e9d7d22
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 384], dtype='float16', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_d05efc18ba61e8edb2f620938f463652(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 196, 384], dtype='float16'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0fb814464ca664f81c94e9aef647bec8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d05efc18ba61e8edb2f620938f463652
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 384], dtype='float16', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3be0823de8a21cc4b10112bb2054c5b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_937f99ba5d70cd6325f7d53e0a367306
    def get_inputs(self):
        return [
            paddle.uniform([1, 25, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_243bda4b16991324f1d0281b6b42595c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 196, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            paddle.static.InputSpec(shape=[320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9007a1d50fa04d2bfc88b0c92ffc53ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_243bda4b16991324f1d0281b6b42595c
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_b8073412405de3409153fccd43d92989(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 197, 768], dtype='float16'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_38c814fcbaea8a15b21b3765d14cebd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8073412405de3409153fccd43d92989
    def get_inputs(self):
        return [
            paddle.uniform([1, 197, 768], dtype='float16', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_b246d33eedb7e4e85c06d4af9d9f5222(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 26, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0af144ad16a653566f9f02b25dfce5fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b246d33eedb7e4e85c06d4af9d9f5222
    def get_inputs(self):
        return [
            paddle.uniform([1, 26, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_50bb534aaf6a4c8cd698cdf1890ba8c1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 200, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e84a52aaddd9df4ef2aaf580dfd567ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50bb534aaf6a4c8cd698cdf1890ba8c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_8849179f5d074ef8df50d7e13dc2683f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 512], dtype='float16'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b5bcf592ec43184bd25bb879dfb78e0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8849179f5d074ef8df50d7e13dc2683f
    def get_inputs(self):
        return [
            paddle.uniform([1, 25, 512], dtype='float16', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_0bd9f48ad5a1ba28632691345c23ee5b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 128], dtype='float16'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ae08b8cc06bbe27ee73e5fd643d26099(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0bd9f48ad5a1ba28632691345c23ee5b
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 128], dtype='float16', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_46f278bf3300d8623c5e81c61377a770(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b4a2b427829fecadb92f3d45377b04b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46f278bf3300d8623c5e81c61377a770
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_1a33f016f927643dc5e246f0362c3b4c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 144], dtype='float32'),
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            paddle.static.InputSpec(shape=[144], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4d36d4dc0ec21d8d742f023718de1f33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a33f016f927643dc5e246f0362c3b4c
    def get_inputs(self):
        return [
            paddle.uniform([4, 256, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_fece71f44bda3a9c3728d72c018f20c2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 384], dtype='float16'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_34785e4a167b9b7c579a8dd92110ff02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fece71f44bda3a9c3728d72c018f20c2
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 384], dtype='float16', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_4e31b4942e66845fa18cfc19a0dc94ad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2083c7051605975db0fb104b0f20d390(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e31b4942e66845fa18cfc19a0dc94ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_5fae025ad456e53ffd0c900e721931d6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 768], dtype='float16'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ad7113ea1d2945d13903f87b5e0a68dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fae025ad456e53ffd0c900e721931d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 768], dtype='float16', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_ed11e7e2542cb43d4b3a4ef724efe0b3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 512], dtype='float16'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2e3a1b114d9633874735522cadceb8d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ed11e7e2542cb43d4b3a4ef724efe0b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 512], dtype='float16', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_25aaecce78402ab576c487feb7a35cb7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 577, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2bc61d442fdf33966af18cf3a6d8e22f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25aaecce78402ab576c487feb7a35cb7
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_771af089da2fc564774a095a0bf6734b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 50, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_97e48df8b215732131f747f2abfca91e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_771af089da2fc564774a095a0bf6734b
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_79422893ac9670c7400ec9a14471c3aa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 256], dtype='float16'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cedff8586305abe4daad0ab667304d5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79422893ac9670c7400ec9a14471c3aa
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 256], dtype='float16', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_84ece0043cb40c018adc993a409dbe42(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 196, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            paddle.static.InputSpec(shape=[320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f9b4b8df54a80bac677389875e07fcc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84ece0043cb40c018adc993a409dbe42
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_4f66f6d8a77466a94167b5fb5b0bd3d8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_85ab3d82e1bf2867806338266e688366(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f66f6d8a77466a94167b5fb5b0bd3d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_fab0a321a75b6de4e9a12bb27282f84f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_95f03321ab43afb15b86a0ae8d7198e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fab0a321a75b6de4e9a12bb27282f84f
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_120a985ec451d79cd6130d6ce016a247(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 197, 768], dtype='float16'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_373e64be0923f3c10d6be646d32901d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_120a985ec451d79cd6130d6ce016a247
    def get_inputs(self):
        return [
            paddle.uniform([1, 197, 768], dtype='float16', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_e3fe96e45a2ce21b2d43587c27e4ccfb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9172a939d9eeb5b33886ccde64d3e5dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3fe96e45a2ce21b2d43587c27e4ccfb
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_a080a38ac7b4d267e648290f6711c8c9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 784, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3d1c9579c3a72604cd34400c30e91027(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a080a38ac7b4d267e648290f6711c8c9
    def get_inputs(self):
        return [
            paddle.uniform([1, 784, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_945573908e2e1a35f58f5888b59e049e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 784, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3193f6791fed5ac6756d31b3e6d3de04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_945573908e2e1a35f58f5888b59e049e
    def get_inputs(self):
        return [
            paddle.uniform([1, 784, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_d7d9c6d92ea8322849caa9afdedf5626(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 577, 768], dtype='float16'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_374705c2cb260a8c321b06df561cf6d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7d9c6d92ea8322849caa9afdedf5626
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 768], dtype='float16', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_b9cb1c401643c1dff62f03931bdd1f80(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 9216, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9691510bd14d036976644682d569861a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9cb1c401643c1dff62f03931bdd1f80
    def get_inputs(self):
        return [
            paddle.uniform([1, 9216, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_0c43932f3ae55dd8d54965e0cc3122b0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3136, 96], dtype='float16'),
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            paddle.static.InputSpec(shape=[96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c519010e36e7eb88cb2cab9e798815a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c43932f3ae55dd8d54965e0cc3122b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 3136, 96], dtype='float16', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_eb68b6cc42aff4c0f4d6edf34ab233ee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 200, 64], dtype='float16'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a263a4aaeec00172cc9579a0a2dfbb84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb68b6cc42aff4c0f4d6edf34ab233ee
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_826bdcbe8204f31a0e9e0400de04a075(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 50, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ea26cc94e15736388466fecdcea01129(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_826bdcbe8204f31a0e9e0400de04a075
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_5966d97bb4ebaea9e5998cfcfc3ce794(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 512], dtype='float16'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0d31cd484019bea74a26a6f98aadb1cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5966d97bb4ebaea9e5998cfcfc3ce794
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 512], dtype='float16', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3306dbb8d6a99e6528974073e4a6cfda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1944912225008011, 0.3826028108596802, 0.16157162189483643, 0.3195856809616089, 0.3475199341773987, 0.1882621943950653, 0.08792823553085327, 0.18206702172756195, 0.10100555419921875, 0.22334177792072296, 0.12090813368558884, 0.048660337924957275, 0.28477728366851807, 0.41904664039611816, 0.4214280843734741, 0.44576936960220337, 0.23108689486980438, 0.16022145748138428, 0.3058986961841583, 0.18708254396915436, 0.3134671747684479, 0.4069231152534485, 0.2430054396390915, 0.2621396780014038], dtype='float32').reshape([24]),
            paddle.to_tensor([0.32334986329078674, 0.3996644616127014, 0.49635884165763855, 0.47662168741226196, 0.4875451326370239, 0.07987800985574722, 0.2242894321680069, 0.30241909623146057, 0.02477540262043476, 0.24675896763801575, 0.33077937364578247, 0.03814496845006943, 0.49355798959732056, 0.45218390226364136, 0.23609976470470428, 0.23890382051467896, 0.2221267819404602, 0.4675355851650238, 0.15177153050899506, 0.1619296371936798, 0.2415061891078949, 0.3584093153476715, 0.17826193571090698, 0.26559266448020935], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_a81fb319cf0944ebabb6b2dc2352deef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_764a0d09955e55d9d63164eada6942be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a81fb319cf0944ebabb6b2dc2352deef
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_b44fdbf136291acd96b634ff68cdefe1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 784, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_adb5ee1637c98d8d9930d2a888a2f11c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b44fdbf136291acd96b634ff68cdefe1
    def get_inputs(self):
        return [
            paddle.uniform([1, 784, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_76ab68a4f64cd208aaed96067e47bd84(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 784, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8712000af22202a45e6527cc6dcd6ee7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76ab68a4f64cd208aaed96067e47bd84
    def get_inputs(self):
        return [
            paddle.uniform([1, 784, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_9762c724605bd2f40bd18d7fccaaf833(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 197, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_77b39f0161c9bdb3369c754427eba03f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9762c724605bd2f40bd18d7fccaaf833
    def get_inputs(self):
        return [
            paddle.uniform([1, 197, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_4e44010dfbba62ca79757686b03cd206(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 784, 192], dtype='float16'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8d9093d93802e2f46fa5987129c2291a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e44010dfbba62ca79757686b03cd206
    def get_inputs(self):
        return [
            paddle.uniform([1, 784, 192], dtype='float16', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_d0d0c8960e5ba257281bc058e06cb5ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3d0c2e6b91a8b89baaf2c0128d0c3563(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d0d0c8960e5ba257281bc058e06cb5ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_6b88fdbcb6d4d141e1c1e8040abc4d42(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 784, 64], dtype='float16'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f34106299165f600c5eb717ebd3d3f6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6b88fdbcb6d4d141e1c1e8040abc4d42
    def get_inputs(self):
        return [
            paddle.uniform([1, 784, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_67fef6d9e76154297e791695a35775e6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 196, 320], dtype='float16'),
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            paddle.static.InputSpec(shape=[320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fffc9730be6594b4c692ece84af3a64b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67fef6d9e76154297e791695a35775e6
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 320], dtype='float16', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_fe1c1cf1fc694579fa16e824e4cfc7f3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 240], dtype='float32'),
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            paddle.static.InputSpec(shape=[240], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_17269d35ff18adb892994be171a7a28e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe1c1cf1fc694579fa16e824e4cfc7f3
    def get_inputs(self):
        return [
            paddle.uniform([4, 16, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_996b58ad674d876e7812f873b8eccb82(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 196, 160], dtype='float16'),
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            paddle.static.InputSpec(shape=[160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b39799f77df00bd199083d35f78a1a0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_996b58ad674d876e7812f873b8eccb82
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 160], dtype='float16', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_6442ebb8a4f6dd1e51ac6b4895ca1ac6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 197, 384], dtype='float16'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8b9902d93743ebcb63aa215da65bdd89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6442ebb8a4f6dd1e51ac6b4895ca1ac6
    def get_inputs(self):
        return [
            paddle.uniform([1, 197, 384], dtype='float16', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_61b9819214790b354856d066b5254fda(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 9216, 128], dtype='float16'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4ccf819d1588d65277ea81156f54acf1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61b9819214790b354856d066b5254fda
    def get_inputs(self):
        return [
            paddle.uniform([1, 9216, 128], dtype='float16', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_9260a9eb5d2a9a519b745defa3dc7af6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 26, 512], dtype='float16'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fd6e8508a84f9776f43334831e43522c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9260a9eb5d2a9a519b745defa3dc7af6
    def get_inputs(self):
        return [
            paddle.uniform([1, 26, 512], dtype='float16', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_add1c87818b8747db67c0ca3be9aabfb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3136, 32], dtype='float16'),
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            paddle.static.InputSpec(shape=[32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_57196d7785f2addc0fe02eefdb1f3d66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_add1c87818b8747db67c0ca3be9aabfb
    def get_inputs(self):
        return [
            paddle.uniform([1, 3136, 32], dtype='float16', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_fc2a09014f97164837a99be3577b88b1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 784, 192], dtype='float16'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9df88732b0ff117196377447cfff70d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc2a09014f97164837a99be3577b88b1
    def get_inputs(self):
        return [
            paddle.uniform([1, 784, 192], dtype='float16', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_b43b09367eb75a46443762c359b6dd41(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 784, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cea316924377eb4bf29f1ee4d339e641(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b43b09367eb75a46443762c359b6dd41
    def get_inputs(self):
        return [
            paddle.uniform([1, 784, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_efe9e779d1a71a6021c7417f96fd76da(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3136, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a596b640ed8e661efde224c0ee93c7fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efe9e779d1a71a6021c7417f96fd76da
    def get_inputs(self):
        return [
            paddle.uniform([1, 3136, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_185a3d2bffd933f33f64b48118189c0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.377737432718277, 0.18530283868312836, 0.14323432743549347, 0.48664844036102295, 0.010780871845781803, 0.4611663222312927, 0.4772086441516876, 0.37828487157821655, 0.1962117701768875, 0.3811570703983307, 0.4023241102695465, 0.28489604592323303, 0.16372263431549072, 0.44700175523757935, 0.25237423181533813, 0.4096340537071228, 0.0903124287724495, 0.04274021461606026, 0.3045683801174164, 0.38907498121261597, 0.0076731350272893906, 0.40988290309906006, 0.26395905017852783, 0.42480334639549255], dtype='float32').reshape([24]),
            paddle.to_tensor([0.042607251554727554, 0.2292732447385788, 0.2175278216600418, 0.4017587900161743, 0.21708501875400543, 0.11139581352472305, 0.12987275421619415, 0.01210138387978077, 0.3472575545310974, 0.3450187146663666, 0.02834395132958889, 0.4251275360584259, 0.2360263168811798, 0.465384304523468, 0.13669751584529877, 0.08593716472387314, 0.023051662370562553, 0.33587512373924255, 0.09113654494285583, 0.3745057284832001, 0.3440038859844208, 0.2501061260700226, 0.15599140524864197, 0.1263386309146881], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 24], dtype='float16'),
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            paddle.static.InputSpec(shape=[24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c99fd7277f8121d947b14975c81fd444(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.38617104291915894, 0.11572497338056564, 0.38189390301704407, 0.16020172834396362, 0.29765796661376953, 0.017672955989837646, 0.03630007058382034, 0.34577760100364685, 0.23633930087089539, 0.294071227312088, 0.4245051443576813, 0.43033868074417114, 0.031015468761324883, 0.2822730243206024, 0.3299134075641632, 0.15210506319999695, 0.49575263261795044, 0.19125495851039886, 0.41863080859184265, 0.3343929052352905, 0.2654281556606293, 0.18332819640636444, 0.1981712132692337, 0.33713534474372864], dtype='float32').reshape([24]),
            paddle.to_tensor([0.316145122051239, 0.27618157863616943, 0.3883530795574188, 0.36047354340553284, 0.4580325484275818, 0.27429041266441345, 0.31387272477149963, 0.31366121768951416, 0.17350850999355316, 0.37955591082572937, 0.03064313903450966, 0.47621357440948486, 0.4102761447429657, 0.3021687865257263, 0.2562055289745331, 0.06368464231491089, 0.3903196156024933, 0.2787001132965088, 0.219029039144516, 0.048097074031829834, 0.2542375922203064, 0.3385213017463684, 0.46876662969589233, 0.019238121807575226], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_4d172ad2aaa384bff102069d7010dace(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 100, 128], dtype='float16'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_94623ce016b9acd19edbdb3dec020faf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d172ad2aaa384bff102069d7010dace
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 128], dtype='float16', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_af7d83e887752f7f27460fc8f59d4743(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.39341840147972107, 0.46687382459640503, 0.10935653001070023, 0.4551464021205902, 0.40478137135505676, 0.3409687578678131, 0.34628233313560486, 0.4930148124694824, 0.4607038199901581, 0.323678582906723, 0.34648799896240234, 0.2895970642566681, 0.35318222641944885, 0.4281359612941742, 0.10394617170095444, 0.2694813013076782, 0.017427900806069374, 0.25487905740737915, 0.292839914560318, 0.4755801856517792, 0.18176168203353882, 0.4752233624458313, 0.21155650913715363, 0.3634304404258728], dtype='float32').reshape([24]),
            paddle.to_tensor([0.4130283296108246, 0.15411455929279327, 0.45077526569366455, 0.45445412397384644, 0.15674017369747162, 0.48320630192756653, 0.3298322558403015, 0.4855938255786896, 0.4683642089366913, 0.4628073573112488, 0.10784704238176346, 0.19944587349891663, 0.3011212646961212, 0.1455358862876892, 0.2366676777601242, 0.3454083204269409, 0.1815304011106491, 0.3239162862300873, 0.2077907770872116, 0.24845445156097412, 0.07662981748580933, 0.4485117197036743, 0.3096381723880768, 0.17961277067661285], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_d75498a38aadad78d5e110232b1b0b05(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 160], dtype='float32'),
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            paddle.static.InputSpec(shape=[160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0a4aae9e024a0c3d7ca2a7f03805dd15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d75498a38aadad78d5e110232b1b0b05
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_0e08952df2309ddf778fd4e3ba94ed04(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6726fc0e42cc37e6738f07b81b887696(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e08952df2309ddf778fd4e3ba94ed04
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_38ce97772e2718429368efa6f84f870d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.25697630643844604, 0.2616826891899109, 0.36537230014801025, 0.13864648342132568, 0.04111538827419281, 0.20468221604824066, 0.3774813413619995, 0.30760225653648376, 0.3268921971321106, 0.16162121295928955, 0.193632110953331, 0.07153741270303726, 0.05428795889019966, 0.3097993731498718, 0.23834088444709778, 0.37003788352012634, 0.04955434054136276, 0.3636513650417328, 0.48130127787590027, 0.425507515668869, 0.07777732610702515, 0.09686791151762009, 0.47500738501548767, 0.17763380706310272], dtype='float32').reshape([24]),
            paddle.to_tensor([0.03983965516090393, 0.011194800958037376, 0.33525604009628296, 0.4724425971508026, 0.22037267684936523, 0.22979885339736938, 0.4378463923931122, 0.3930085599422455, 0.2661578953266144, 0.06824540346860886, 0.30339109897613525, 0.2867439389228821, 0.060707103461027145, 0.3062986433506012, 0.29465994238853455, 0.46912169456481934, 0.025963183492422104, 0.25091734528541565, 0.042882289737463, 0.4332965910434723, 0.1658511757850647, 0.15828007459640503, 0.11682777106761932, 0.12158512324094772], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_fd76dde7897eb5024349a84835b9a895(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, 1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fef4c9111d47c30eaa0cb1559aea643b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd76dde7897eb5024349a84835b9a895
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_ca64d98122be9855566edd7ce39d6f38(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 100, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_13a6f3387658e413f005155897945851(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca64d98122be9855566edd7ce39d6f38
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_98076a412c77a91daadd5b6e87472d20(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            paddle.static.InputSpec(shape=[320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fffb331858a3b2f0a0efd546116408f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98076a412c77a91daadd5b6e87472d20
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_d07565080e76228d3518844748f235d9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 512], dtype='float16'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d00b2d7e52808baef8e8b8296c16565f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d07565080e76228d3518844748f235d9
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4ba70ec91fc9e171bb3ec811f687f3ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2218901515007019, 0.36188772320747375, 0.04948639124631882, 0.47907280921936035, 0.19810333847999573, 0.2595363259315491, 0.2106567919254303, 0.42106103897094727, 0.13477306067943573, 0.13146597146987915, 0.08650773018598557, 0.07896021008491516, 0.1654050201177597, 0.08941850066184998, 0.09084242582321167, 0.18809713423252106, 0.46906110644340515, 0.14247792959213257, 0.07266838103532791, 0.4778778851032257, 0.16100482642650604, 0.3071631193161011, 0.30422717332839966, 0.019381634891033173], dtype='float32').reshape([24]),
            paddle.to_tensor([0.003561025485396385, 0.46045517921447754, 0.21475692093372345, 0.010452022776007652, 0.3242853879928589, 0.2562505602836609, 0.33443042635917664, 0.48823460936546326, 0.48607632517814636, 0.33735978603363037, 0.4287622570991516, 0.28406837582588196, 0.24270394444465637, 0.14426647126674652, 0.10508942604064941, 0.10875041037797928, 0.02983558550477028, 0.4991454482078552, 0.1449068933725357, 0.2402506321668625, 0.3209373354911804, 0.06510025262832642, 0.0939621850848198, 0.3456525504589081], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_60517ba961807371eabb8b5c57c07c29(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 50, 256], dtype='float16'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_234c3cacbf037368a437f9a0656f5405(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60517ba961807371eabb8b5c57c07c29
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 256], dtype='float16', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_1db26a12080162c71898598c11e7c804(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dba234e3449ed32e0cee53b18cd784e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1db26a12080162c71898598c11e7c804
    def get_inputs(self):
        return [
            paddle.uniform([4, 64, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4dba6fb641fa68c651d516c9aefe8c73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.21724389493465424, 0.21386022865772247, 0.33139729499816895, 0.42786169052124023, 0.23576483130455017, 0.39199674129486084, 0.4420982897281647, 0.18765150010585785, 0.40616729855537415, 0.2982950210571289, 0.030321791768074036, 0.023114165291190147, 0.03075837902724743, 0.3489494025707245, 0.31738221645355225, 0.3437778055667877, 0.3625026345252991, 0.42859336733818054, 0.2719637453556061, 0.17685560882091522, 0.21495655179023743, 0.1715925931930542, 0.433343768119812, 0.4940958321094513], dtype='float32').reshape([24]),
            paddle.to_tensor([0.12360720336437225, 0.47353073954582214, 0.2214878648519516, 0.16142845153808594, 0.14972840249538422, 0.03269874304533005, 0.3545283377170563, 0.4615774154663086, 0.012213196605443954, 0.012177485972642899, 0.07326292246580124, 0.45970240235328674, 0.04811980947852135, 0.3012188971042633, 0.07776078581809998, 0.11717414855957031, 0.2512454390525818, 0.44061145186424255, 0.12461359798908234, 0.4159942865371704, 0.2479582130908966, 0.26588964462280273, 0.37404027581214905, 0.3919767439365387], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_948579d48ed42627af97258c6d530789(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 240], dtype='float16'),
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            paddle.static.InputSpec(shape=[240], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e422825340eb44d98f32080678ca09b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_948579d48ed42627af97258c6d530789
    def get_inputs(self):
        return [
            paddle.uniform([4, 16, 240], dtype='float16', min=0, max=0.5),
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_351d71e9591bcb7783c07e4f4f380d17(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3136, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            paddle.static.InputSpec(shape=[96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1ffca9908b8f4b3f20889fccbd9af31b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_351d71e9591bcb7783c07e4f4f380d17
    def get_inputs(self):
        return [
            paddle.uniform([1, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_0c1bc01a9019f47f2a80aabf0c65dc2b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2304, 512], dtype='float16'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_60519f6399dd37c86ed426b5ab6ec1ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c1bc01a9019f47f2a80aabf0c65dc2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 2304, 512], dtype='float16', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_dd1a8ccd6ce65a97d9c5683f556229f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_09bb260a1b92580bcacd7fe4d4aac712(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd1a8ccd6ce65a97d9c5683f556229f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_e65c2b3383197e131b1ab0ffd13efb05(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3136, 64], dtype='float16'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_682ef60e5d0343b62f44125fccf9c12b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e65c2b3383197e131b1ab0ffd13efb05
    def get_inputs(self):
        return [
            paddle.uniform([1, 3136, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_75e7e64732c7c6ccf15cd528dfd4fef9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3136, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            paddle.static.InputSpec(shape=[96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_496d1e80f8dfe3ae82879f7cc1bbc71a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75e7e64732c7c6ccf15cd528dfd4fef9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_3d62404dbd033bdc163e060c3ce9a389(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2304, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e3623517c364a51de478dd9fb247070a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d62404dbd033bdc163e060c3ce9a389
    def get_inputs(self):
        return [
            paddle.uniform([1, 2304, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_44355986b76d9d1ec744346261b82e00(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 160], dtype='float16'),
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            paddle.static.InputSpec(shape=[160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4c2b3b38a7a0a0901c74d1ed0acdaf00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44355986b76d9d1ec744346261b82e00
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 160], dtype='float16', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_8c271728cf0f315e87b62e614a45b46d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 144], dtype='float16'),
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            paddle.static.InputSpec(shape=[144], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_07a562da43fb637acebe7fc154f96556(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c271728cf0f315e87b62e614a45b46d
    def get_inputs(self):
        return [
            paddle.uniform([4, 256, 144], dtype='float16', min=0, max=0.5),
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_656d73e975ded93491678fc0452b556b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 576, 1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b3fa6e79764adc3626ad7ca452a7a8c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_656d73e975ded93491678fc0452b556b
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_ff6c86c1e8a2353c0da603ba5beb31fd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 768], dtype='float16'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5b26a731b12d9df778a7b91c70b872c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff6c86c1e8a2353c0da603ba5beb31fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 768], dtype='float16', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5fc9f33f626a0a911121432b413e1ed5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2948295474052429, 0.25656041502952576, 0.02464757114648819, 0.476041316986084, 0.19090557098388672, 0.4085657298564911, 0.49969279766082764, 0.3506554961204529, 0.3696248233318329, 0.04948889836668968, 0.07433341443538666, 0.4906398057937622, 0.3546501100063324, 0.1728094220161438, 0.2735897898674011, 0.47032979130744934, 0.46880200505256653, 0.277831107378006, 0.3094508647918701, 0.3404964506626129, 0.06734950095415115, 0.022301096469163895, 0.235468327999115, 0.4539664387702942], dtype='float32').reshape([24]),
            paddle.to_tensor([0.22101391851902008, 0.07503221929073334, 0.398092120885849, 0.4442005753517151, 0.334077388048172, 0.14161737263202667, 0.29515066742897034, 0.15676969289779663, 0.07603178173303604, 0.2450062781572342, 0.06768611818552017, 0.04412431642413139, 0.15744595229625702, 0.19754409790039062, 0.35324999690055847, 0.015365750528872013, 0.10068827122449875, 0.38528066873550415, 0.4668484330177307, 0.22386804223060608, 0.3475293219089508, 0.23377925157546997, 0.037336625158786774, 0.3932895362377167], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_80d1bff3499b380e40b9d4e8d64755b5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, 1024], dtype='float16'),
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2e97dae47081523a290116c1504e921a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80d1bff3499b380e40b9d4e8d64755b5
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1024], dtype='float16', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_124a7efc51219f8dc1dec25adb89b860(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 784, 128], dtype='float16'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4f624fd8eaab11720d31f91cacd8bc22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_124a7efc51219f8dc1dec25adb89b860
    def get_inputs(self):
        return [
            paddle.uniform([1, 784, 128], dtype='float16', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_188986d6741f24baaf0a3d19b5e200bf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 256], dtype='float16'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_75ecd1c27ada6a8bd84b094da74a95eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_188986d6741f24baaf0a3d19b5e200bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 256], dtype='float16', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_d62776885a616123801c290bc97a8b3c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 100, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_57e3eb3702c2487d416f73b7b4d3bce1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d62776885a616123801c290bc97a8b3c
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_86c40ede5ca74b70f3666f068e562bfd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 50, 256], dtype='float16'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2e519f2a063f62d63f5de578ecfa07ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86c40ede5ca74b70f3666f068e562bfd
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 256], dtype='float16', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_7cdfab677827819afac50a8c90c94fbe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3136, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            paddle.static.InputSpec(shape=[32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f26829d6ef511f96cbaa7c1e688f0925(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cdfab677827819afac50a8c90c94fbe
    def get_inputs(self):
        return [
            paddle.uniform([1, 3136, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_265c0e57c8bae7fb96af23f74fdea015(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4075280427932739, 0.28762948513031006, 0.14922772347927094, 0.16289710998535156, 0.36465883255004883, 0.06658970564603806, 0.3865613043308258, 0.037496376782655716, 0.4020816385746002, 0.38956156373023987, 0.03481167182326317, 0.0018752855248749256, 0.07107016444206238, 0.13412366807460785, 0.34783926606178284, 0.2869226336479187, 0.3445194363594055, 0.27523764967918396, 0.3691956102848053, 0.23720872402191162, 0.10267768800258636, 0.16095612943172455, 0.0019124711398035288, 0.46512287855148315], dtype='float32').reshape([24]),
            paddle.to_tensor([0.08091019093990326, 0.04433393105864525, 0.058001041412353516, 0.001132540637627244, 0.2740916907787323, 0.38383543491363525, 0.08343417197465897, 0.13556107878684998, 0.018043357878923416, 0.49788594245910645, 0.38847774267196655, 0.24051205813884735, 0.09503314644098282, 0.29158028960227966, 0.4197055995464325, 0.30534827709198, 0.49423396587371826, 0.19452549517154694, 0.09706449508666992, 0.2676755487918854, 0.4134049713611603, 0.16105583310127258, 0.14482448995113373, 0.30570554733276367], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8fc17d4800b552c5b9939d3d1918c2d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.11138676851987839, 0.47739094495773315, 0.1473296582698822, 0.23400244116783142, 0.2873220145702362, 0.3124854266643524, 0.40942201018333435, 0.34808632731437683, 0.04339507594704628, 0.26135051250457764, 0.05586106330156326, 0.4238629937171936, 0.4491181969642639, 0.4265049397945404, 0.4986898899078369, 0.417705237865448, 0.48625999689102173, 0.3012504577636719, 0.45965704321861267, 0.15417903661727905, 0.10371867567300797, 0.43387138843536377, 0.31557005643844604, 0.03589140251278877], dtype='float32').reshape([24]),
            paddle.to_tensor([0.42013609409332275, 0.42081624269485474, 0.15403145551681519, 0.16094908118247986, 0.3998282551765442, 0.02676352486014366, 0.0278155617415905, 0.4778689444065094, 0.449493408203125, 0.4048968255519867, 0.37809595465660095, 0.08318862318992615, 0.20823676884174347, 0.25754427909851074, 0.35324618220329285, 0.0029195775277912617, 0.29552334547042847, 0.4850471317768097, 0.4745393991470337, 0.35328713059425354, 0.011256619356572628, 0.4099480211734772, 0.3874702453613281, 0.31143122911453247], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_032235d3bbb8d048478a35ac82cc344b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 64], dtype='float16'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4f49231131dc1acd6075854902b39ef9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_032235d3bbb8d048478a35ac82cc344b
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_aac44a73b00bf30ed87610d317d20ec9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, 2048], dtype='float32'),
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6a9f6993a7571d41e28b318525abcca2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aac44a73b00bf30ed87610d317d20ec9
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 2048], dtype='float32', min=0, max=0.5),
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_5d90f9210ad2f5e7f4099981446e8fce(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            paddle.static.InputSpec(shape=[96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ee317ff3499c59b5ee90605b121ba48d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d90f9210ad2f5e7f4099981446e8fce
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_5c53405c45b63972cf6b8f355c048412(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 192], dtype='float16'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6f3ca6dd284f49f091946e9745769682(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c53405c45b63972cf6b8f355c048412
    def get_inputs(self):
        return [
            paddle.uniform([4, 64, 192], dtype='float16', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_0aa9e55029dab4942371cb2922fb3ab0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2304, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9826b58bbc6b2b1baae545d8f6130a02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0aa9e55029dab4942371cb2922fb3ab0
    def get_inputs(self):
        return [
            paddle.uniform([1, 2304, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_85baa503704e8da162fd2b603960b0db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2733425498008728, 0.2886544466018677, 0.12506115436553955, 0.2635963261127472, 0.40912050008773804, 0.14918892085552216, 0.36996743083000183, 0.40817272663116455, 0.3628562390804291, 0.022754129022359848, 0.03988342732191086, 0.30717965960502625, 0.47920846939086914, 0.183665931224823, 0.3315567076206207, 0.3608635663986206, 0.030695747584104538, 0.11753777414560318, 0.015341474674642086, 0.3680242598056793, 0.2977781295776367, 0.42913395166397095, 0.10832520574331284, 0.28080159425735474], dtype='float32').reshape([24]),
            paddle.to_tensor([0.3170471489429474, 0.22752729058265686, 0.3965907394886017, 0.47981470823287964, 0.34056150913238525, 0.45482271909713745, 0.10854624211788177, 0.44269371032714844, 0.0010614944621920586, 0.3491353690624237, 0.37615808844566345, 0.2246566265821457, 0.33170726895332336, 0.2919798493385315, 0.06577467173337936, 0.13128428161144257, 0.3963892459869385, 0.1373082846403122, 0.087900310754776, 0.3702240586280823, 0.029022011905908585, 0.16377879679203033, 0.4532797038555145, 0.25386375188827515], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_661e8e5e7dc8cf8b5e8dfe507119b420(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4942370355129242, 0.24279873073101044, 0.07200321555137634, 0.19086234271526337, 0.37410399317741394, 0.4693959653377533, 0.42202505469322205, 0.3212796449661255, 0.3400349020957947, 0.033894605934619904, 0.4438764750957489, 0.0531991608440876, 0.29941052198410034, 0.37350213527679443, 0.3689129054546356, 0.2065431773662567, 0.2327260971069336, 0.34288451075553894, 0.2392301708459854, 0.04949093237519264, 0.14563585817813873, 0.45946016907691956, 0.30910953879356384, 0.4555950462818146], dtype='float32').reshape([24]),
            paddle.to_tensor([0.4361044466495514, 0.2024260312318802, 0.04483854025602341, 0.23142854869365692, 0.17983077466487885, 0.21591337025165558, 0.45661216974258423, 0.12247885018587112, 0.005528866779059172, 0.01618219166994095, 0.3145701289176941, 0.21036936342716217, 0.2469642609357834, 0.39734336733818054, 0.14492493867874146, 0.15792416036128998, 0.29649606347084045, 0.24214282631874084, 0.051815465092659, 0.3278498947620392, 0.37583300471305847, 0.0743638202548027, 0.4275074899196625, 0.457813560962677], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_4a3e458f3bc9adb63425d460898264eb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 196, 160], dtype='float32'),
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            paddle.static.InputSpec(shape=[160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ebaaa18c887d9bbafbf4595bc23a15ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a3e458f3bc9adb63425d460898264eb
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_8dd42229734a93b51f6df67a38866d14(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 196, 768], dtype='float16'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c33f2b302021795042d84cce791b0133(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8dd42229734a93b51f6df67a38866d14
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 768], dtype='float16', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_eee3955791de9461dd0ba26a54ba6149(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 192], dtype='float16'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_45a59321492d9595e46dc246722499b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eee3955791de9461dd0ba26a54ba6149
    def get_inputs(self):
        return [
            paddle.uniform([4, 64, 192], dtype='float16', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_92ce833570191359ba1946438e0ec3a2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 100, 128], dtype='float16'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_57704545ec2ecfcc3bdfc25bde0aa28a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92ce833570191359ba1946438e0ec3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 128], dtype='float16', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_1cb3f21028b66566615e879ed67fa628(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3136, 96], dtype='float16'),
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            paddle.static.InputSpec(shape=[96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3a3142ac9878643354485e1334978d0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1cb3f21028b66566615e879ed67fa628
    def get_inputs(self):
        return [
            paddle.uniform([1, 3136, 96], dtype='float16', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_22b70ff14db9e9b39dd8dba72490b286(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2749462425708771, 0.1498604416847229, 0.3775080144405365, 0.42425140738487244, 0.49972760677337646, 0.24153651297092438, 0.35319769382476807, 0.14112374186515808, 0.24404770135879517, 0.4066070020198822, 0.18401911854743958, 0.40190380811691284, 0.34167197346687317, 0.296133816242218, 0.44441258907318115, 0.4292530119419098, 0.34508177638053894, 0.4548507034778595, 0.4349629878997803, 0.15553048253059387, 0.48401716351509094, 0.2851331830024719, 0.48180171847343445, 0.19099731743335724], dtype='float32').reshape([24]),
            paddle.to_tensor([0.08021919429302216, 0.14500804245471954, 0.2378561794757843, 0.1521240621805191, 0.0067772469483315945, 0.41196808218955994, 0.26628732681274414, 0.18496370315551758, 0.040576208382844925, 0.33428266644477844, 0.3456243872642517, 0.3837692141532898, 0.2483023852109909, 0.0924777090549469, 0.18694263696670532, 0.0643448531627655, 0.2472136914730072, 0.1790827065706253, 0.3531460464000702, 0.2753889858722687, 0.24934150278568268, 0.3769308924674988, 0.12358508259057999, 0.1369844228029251], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_96d1ee1fb3ae0e22e962da3414e565ee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eb39d5dfbf86af5f748b3665d381dbae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96d1ee1fb3ae0e22e962da3414e565ee
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_0a0e8b036a643006abac343bacb204e0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3136, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            paddle.static.InputSpec(shape=[32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_47145eb9f17e8ec62235966e58c195a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a0e8b036a643006abac343bacb204e0
    def get_inputs(self):
        return [
            paddle.uniform([1, 3136, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_988168fcdf8b8324162815ee64e1f07b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.11822105199098587, 0.35403379797935486, 0.09216026216745377, 0.4268139600753784, 0.3254181742668152, 0.10379862040281296, 0.2708767056465149, 0.45560401678085327, 0.1446813941001892, 0.025853773579001427, 0.4610397219657898, 0.10393553972244263, 0.25771042704582214, 0.34118756651878357, 0.43695932626724243, 0.2615312933921814, 0.4114696681499481, 0.20956392586231232, 0.4371609091758728, 0.23120185732841492, 0.4584100544452667, 0.06640075892210007, 0.46699827909469604, 0.26636576652526855], dtype='float32').reshape([24]),
            paddle.to_tensor([0.41902992129325867, 0.16590248048305511, 0.3327663242816925, 0.42059892416000366, 0.09136369079351425, 0.43001842498779297, 0.15384966135025024, 0.299831360578537, 0.17848137021064758, 0.22327029705047607, 0.19837695360183716, 0.034178365021944046, 0.1374148726463318, 0.12881284952163696, 0.13214822113513947, 0.4897172749042511, 0.1868322193622589, 0.48641303181648254, 0.07325220853090286, 0.1059931293129921, 0.18990863859653473, 0.10014212876558304, 0.26787856221199036, 0.3915049135684967], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_7cb413a72c1ae67d0ece2c769e1468be(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 784, 64], dtype='float16'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9c5d928e6dd8f5b2214243381000394b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cb413a72c1ae67d0ece2c769e1468be
    def get_inputs(self):
        return [
            paddle.uniform([1, 784, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b2f04109704c894b2ce894712b983741(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.20196907222270966, 0.3077658414840698, 0.30164971947669983, 0.3887242376804352, 0.2066408097743988, 0.1929297298192978, 0.42979133129119873, 0.3063250780105591, 0.15751884877681732, 0.29275304079055786, 0.3532087802886963, 0.20203478634357452, 0.32315364480018616, 0.16049759089946747, 0.16638819873332977, 0.383476197719574, 0.49989399313926697, 0.4861551523208618, 0.3152659237384796, 0.08918523788452148, 0.2502291202545166, 0.4488310217857361, 0.26767122745513916, 0.21805009245872498], dtype='float32').reshape([24]),
            paddle.to_tensor([0.44470223784446716, 0.2137417495250702, 0.03797849640250206, 0.0681903213262558, 0.24024896323680878, 0.3687047064304352, 0.30915430188179016, 0.06465546786785126, 0.21601150929927826, 0.07825257629156113, 0.3345701992511749, 0.12046723067760468, 0.33100271224975586, 0.20086514949798584, 0.17381206154823303, 0.14239880442619324, 0.3918207585811615, 0.4170031249523163, 0.29186883568763733, 0.239900141954422, 0.33178573846817017, 0.23659555613994598, 0.2637886703014374, 0.33635568618774414], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dd52c1abf4b7dfbcf60d98ed0a687bb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.21603699028491974, 0.457147479057312, 0.32761427760124207, 0.43386441469192505, 0.3132554292678833, 0.12883122265338898, 0.34900692105293274, 0.15262621641159058, 0.14229939877986908, 0.0941244587302208, 0.13210459053516388, 0.07716771215200424, 0.17410129308700562, 0.42766106128692627, 0.28403162956237793, 0.45614659786224365, 0.022199802100658417, 0.2599000930786133, 0.11217594146728516, 0.3731764554977417, 0.2556445002555847, 0.1427105963230133, 0.28403982520103455, 0.18033753335475922], dtype='float32').reshape([24]),
            paddle.to_tensor([0.08212942630052567, 0.4710656702518463, 0.4386807978153229, 0.43541672825813293, 0.1459144949913025, 0.016623863950371742, 0.2603388726711273, 0.2934587001800537, 0.4624547064304352, 0.21479105949401855, 0.22284936904907227, 0.26773396134376526, 0.2879648208618164, 0.22528375685214996, 0.3968689441680908, 0.41876912117004395, 0.3529714345932007, 0.29852938652038574, 0.34880825877189636, 0.3127513527870178, 0.25624874234199524, 0.30797886848449707, 0.21493294835090637, 0.2662631869316101], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_6a868288f22e057105c6a7169fd949a5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, 2048], dtype='float16'),
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4292c669d339aa703f96afdd43801532(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a868288f22e057105c6a7169fd949a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 2048], dtype='float16', min=0, max=0.5),
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_1dab2a3c19dc0534e9ec6db315e8a9d9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 784, 128], dtype='float16'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_aa3acdbb12e2d4425489391767df80df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1dab2a3c19dc0534e9ec6db315e8a9d9
    def get_inputs(self):
        return [
            paddle.uniform([1, 784, 128], dtype='float16', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_c975aced4862620259d590acb0b59e57(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 144], dtype='float16'),
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            paddle.static.InputSpec(shape=[144], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c33a486797bc534f7a35da0ef38d140e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c975aced4862620259d590acb0b59e57
    def get_inputs(self):
        return [
            paddle.uniform([4, 256, 144], dtype='float16', min=0, max=0.5),
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_41cd05d31cc9a73ae8b5261a24a032dd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            paddle.static.InputSpec(shape=[32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cb848613d6366a316f13bb4b83db18ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41cd05d31cc9a73ae8b5261a24a032dd
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b61dfdc61dab89026029355cd752e991(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.13104449212551117, 0.001003191340714693, 0.0004518513451330364, 0.3053096830844879, 0.4612765312194824, 0.2768879532814026, 0.09664745628833771, 0.22984181344509125, 0.3971695303916931, 0.26662760972976685, 0.03428388014435768, 0.2252470850944519, 0.16367743909358978, 0.22451382875442505, 0.4354277551174164, 0.18384778499603271, 0.2191624492406845, 0.34630006551742554, 0.1290626972913742, 0.03872497007250786, 0.03579448163509369, 0.37888261675834656, 0.09405921399593353, 0.19688816368579865], dtype='float32').reshape([24]),
            paddle.to_tensor([0.22849002480506897, 0.22476203739643097, 0.2837699353694916, 0.1893600970506668, 0.23464249074459076, 0.29270437359809875, 0.043193090707063675, 0.17402663826942444, 0.20343337953090668, 0.13933992385864258, 0.08156461268663406, 0.2924439609050751, 0.39190566539764404, 0.001491321949288249, 0.2799133360385895, 0.014528901316225529, 0.45894384384155273, 0.3823985159397125, 0.4985741972923279, 0.1520860642194748, 0.07000112533569336, 0.43293774127960205, 0.475347638130188, 0.21308377385139465], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9b4e99c0efe5be99ded5880a6f319b5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.08189158141613007, 0.2640123665332794, 0.40644341707229614, 0.38590019941329956, 0.3514549434185028, 0.1318885236978531, 0.015498715452849865, 0.10658221691846848, 0.4806549549102783, 0.094437375664711, 0.07244237512350082, 0.4244176149368286, 0.1667337417602539, 0.30966898798942566, 0.33457323908805847, 0.1593007892370224, 0.2961292266845703, 0.05713701248168945, 0.43964460492134094, 0.12787842750549316, 0.15702253580093384, 0.022814583033323288, 0.18830177187919617, 0.2869359850883484], dtype='float32').reshape([24]),
            paddle.to_tensor([0.17696243524551392, 0.20154716074466705, 0.35040247440338135, 0.03410183638334274, 0.1347581446170807, 0.12588974833488464, 0.33610716462135315, 0.17126868665218353, 0.15495172142982483, 0.3760520815849304, 0.07632195204496384, 0.32297757267951965, 0.25233492255210876, 0.15374934673309326, 0.46340006589889526, 0.47967272996902466, 0.4464777410030365, 0.22481876611709595, 0.02110927365720272, 0.39305034279823303, 0.21820755302906036, 0.24352051317691803, 0.07854853570461273, 0.09641434997320175], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_ac73b851feaab6eec671131b794cc11e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 196, 160], dtype='float32'),
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            paddle.static.InputSpec(shape=[160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5b96ed047c744ebcd3e7e4433b29887b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac73b851feaab6eec671131b794cc11e
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cf980234b7a7c2390c78e79632f78343(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3032447397708893, 0.2310774326324463, 0.3592204749584198, 0.2898314595222473, 0.4603426456451416, 0.17551076412200928, 0.15908095240592957, 0.05398900806903839, 0.3695704936981201, 0.4119839370250702, 0.2586297392845154, 0.011514166370034218, 0.3243766725063324, 0.415735125541687, 0.4045689105987549, 0.1378740668296814, 0.12950938940048218, 0.06598066538572311, 0.0520794577896595, 0.4184545874595642, 0.019061435014009476, 0.15277910232543945, 0.46421897411346436, 0.07483349740505219], dtype='float32').reshape([24]),
            paddle.to_tensor([0.3027246296405792, 0.18151690065860748, 0.13708433508872986, 0.2975739538669586, 0.15942570567131042, 0.19100406765937805, 0.4496937394142151, 0.1999097466468811, 0.34864258766174316, 0.07857060432434082, 0.12883058190345764, 0.37036246061325073, 0.4196438193321228, 0.48509618639945984, 0.4795878231525421, 0.35608184337615967, 0.15370695292949677, 0.09583970904350281, 0.0893140584230423, 0.4857872724533081, 0.08927138894796371, 0.14895647764205933, 0.061342813074588776, 0.23118531703948975], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_4e4a9e383b7df49d051e6b8c5b585349(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2304, 256], dtype='float16'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7fdc4504ea4caf3421f698ed87c2d1ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e4a9e383b7df49d051e6b8c5b585349
    def get_inputs(self):
        return [
            paddle.uniform([1, 2304, 256], dtype='float16', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bb2a7af9476f3d3fdd21c2a38de0ad4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.003355184569954872, 0.08527450263500214, 0.49113503098487854, 0.06987988203763962, 0.3817385137081146, 0.057965606451034546, 0.28964295983314514, 0.37287047505378723, 0.20860333740711212, 0.060067348182201385, 0.2851361036300659, 0.12783615291118622, 0.07190290838479996, 0.40754643082618713, 0.36866819858551025, 0.20896939933300018, 0.21901348233222961, 0.16887909173965454, 0.2644297182559967, 0.303053081035614, 0.1525713950395584, 0.06794065237045288, 0.2204035073518753, 0.30499154329299927], dtype='float32').reshape([24]),
            paddle.to_tensor([0.4337674379348755, 0.12172119319438934, 0.33910325169563293, 0.2982933819293976, 0.3377218544483185, 0.10523975640535355, 0.29910847544670105, 0.2660064697265625, 0.1492518037557602, 0.004364924039691687, 0.43609419465065, 0.3189098536968231, 0.04427649453282356, 0.37041908502578735, 0.1509520411491394, 0.037587255239486694, 0.3917332589626312, 0.24273355305194855, 0.38657689094543457, 0.04726611077785492, 0.41798269748687744, 0.25149938464164734, 0.1313762068748474, 0.017108848318457603], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_5f9d11a6ca1b1451d1023ece1eca04b3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 196, 160], dtype='float16'),
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            paddle.static.InputSpec(shape=[160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_20f6bbfd9e2e906ebffbdbff00d4f46a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f9d11a6ca1b1451d1023ece1eca04b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 160], dtype='float16', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a666c711d8577773f86c7bfbd3374ed5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.22059546411037445, 0.08107399195432663, 0.3599371910095215, 0.06091917306184769, 0.13248299062252045, 0.48933812975883484, 0.35126540064811707, 0.40275704860687256, 0.3335936367511749, 0.4690301716327667, 0.10812464356422424, 0.21098940074443817, 0.3509853184223175, 0.08305541425943375, 0.29790058732032776, 0.28631922602653503, 0.41525518894195557, 0.19078561663627625, 0.30999377369880676, 0.17886129021644592, 0.3645116090774536, 0.30231353640556335, 0.27994534373283386, 0.4795728623867035], dtype='float32').reshape([24]),
            paddle.to_tensor([0.49116116762161255, 0.2589567303657532, 0.21399252116680145, 0.4345894157886505, 0.06589052826166153, 0.032483458518981934, 0.4032805562019348, 0.2811431586742401, 0.18462687730789185, 0.3623270094394684, 0.3536543548107147, 0.32368308305740356, 0.24250781536102295, 0.32686877250671387, 0.17247992753982544, 0.3505391776561737, 0.019868867471814156, 0.07206461578607559, 0.17273980379104614, 0.18356630206108093, 0.45990994572639465, 0.47260043025016785, 0.11956190317869186, 0.44359534978866577], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_9fe5a14c8756c432c3c85131da9acc90(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3136, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fcba6a4ce9bb6abcd449b03d3fd4981a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9fe5a14c8756c432c3c85131da9acc90
    def get_inputs(self):
        return [
            paddle.uniform([1, 3136, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_360d3fd18a063486bb3c410714aa1d29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.07777924090623856, 0.1877835988998413, 0.44796645641326904, 0.13612505793571472, 0.44554564356803894, 0.19828172028064728, 0.4584275484085083, 0.48967087268829346, 0.27936676144599915, 0.12289503216743469, 0.17118236422538757, 0.02821151539683342, 0.32624655961990356, 0.22541551291942596, 0.47783684730529785, 0.3456783592700958, 0.18602311611175537, 0.06742773205041885, 0.18299618363380432, 0.48287785053253174, 0.3348250389099121, 0.44649025797843933, 0.36535608768463135, 0.3382311761379242], dtype='float32').reshape([24]),
            paddle.to_tensor([0.09664707630872726, 0.27151766419410706, 0.46327680349349976, 0.22908218204975128, 0.46835148334503174, 0.43873128294944763, 0.04229975864291191, 0.4763283431529999, 0.3355560004711151, 0.0445965901017189, 0.44537651538848877, 0.07344529777765274, 0.030793532729148865, 0.2414197474718094, 0.45863765478134155, 0.3339186906814575, 0.2636232376098633, 0.19587001204490662, 0.4803667366504669, 0.26928606629371643, 0.23129412531852722, 0.05639997124671936, 0.4081173837184906, 0.26974520087242126], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_0ea392b61ab64b9e90e6cf691bc44b23(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 784, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1df6378c570481bbe6c03a3c5bc3eb11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ea392b61ab64b9e90e6cf691bc44b23
    def get_inputs(self):
        return [
            paddle.uniform([1, 784, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e7031cb5a54ae874ed558d455ece1bb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4821987450122833, 0.265827476978302, 0.35885751247406006, 0.4651005268096924, 0.2871500849723816, 0.09241783618927002, 0.23046033084392548, 0.2547864615917206, 0.2563897967338562, 0.44208091497421265, 0.27567529678344727, 0.03303602337837219, 0.02879159152507782, 0.29305702447891235, 0.43727824091911316, 0.09752491116523743, 0.1959545612335205, 0.09608925879001617, 0.33840739727020264, 0.3394774794578552, 0.362621933221817, 0.361385315656662, 0.2533928155899048, 0.1054086685180664], dtype='float32').reshape([24]),
            paddle.to_tensor([0.2189566045999527, 0.1663825958967209, 0.3713485300540924, 0.4094517230987549, 0.1551073044538498, 0.41604241728782654, 0.13740676641464233, 0.006408956367522478, 0.3149452805519104, 0.2914600074291229, 0.303353488445282, 0.0042930361814796925, 0.21233555674552917, 0.2439674288034439, 0.3981761634349823, 0.31534624099731445, 0.17092184722423553, 0.111934132874012, 0.1160784587264061, 0.4051275849342346, 0.2443031668663025, 0.26640191674232483, 0.14770035445690155, 0.2985193431377411], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e69c70abbf8f7574dcc3e5394b8bc9b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3569045960903168, 0.4219363033771515, 0.3463732600212097, 0.24545809626579285, 0.15596434473991394, 0.4199153184890747, 0.1694270521402359, 0.010177422314882278, 0.27669385075569153, 0.2522677481174469, 0.05690763145685196, 0.0931222066283226, 0.07311534881591797, 0.39175882935523987, 0.39445561170578003, 0.10012450814247131, 0.22885538637638092, 0.3355445861816406, 0.3250260055065155, 0.16773413121700287, 0.16229140758514404, 0.39997437596321106, 0.2962440550327301, 0.2199307531118393], dtype='float32').reshape([24]),
            paddle.to_tensor([0.19937656819820404, 0.2022133618593216, 0.4590785801410675, 0.07124201208353043, 0.2472957819700241, 0.253879189491272, 0.045171111822128296, 0.2538730204105377, 0.177532359957695, 0.31650349497795105, 0.004671516362577677, 0.23363719880580902, 0.17990192770957947, 0.21386675536632538, 0.31672754883766174, 0.442676305770874, 0.06987858563661575, 0.3340548872947693, 0.14056098461151123, 0.2355816662311554, 0.38369229435920715, 0.12451627105474472, 0.3545750081539154, 0.38873907923698425], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ff98862134c542046a4e555f51c1e9a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.35858413577079773, 0.34788861870765686, 0.2857751250267029, 0.25739121437072754, 0.49368444085121155, 0.3644040524959564, 0.1073170006275177, 0.18009315431118011, 0.19283881783485413, 0.4494086503982544, 0.1005583107471466, 0.445462703704834, 0.19405803084373474, 0.4874476492404938, 0.08407430350780487, 0.2414054274559021, 0.2729944884777069, 0.3681156635284424, 0.09988174587488174, 0.08254747092723846, 0.2911469638347626, 0.3688807189464569, 0.48850077390670776, 0.12959858775138855], dtype='float32').reshape([24]),
            paddle.to_tensor([0.289671927690506, 0.4417371451854706, 0.45394620299339294, 0.038958366960287094, 0.39196792244911194, 0.31908389925956726, 0.3375179171562195, 0.43845829367637634, 0.4918230473995209, 0.2424422949552536, 0.04904443025588989, 0.27253976464271545, 0.252619206905365, 0.23734422028064728, 0.23768794536590576, 0.3596724569797516, 0.3481784462928772, 0.30672597885131836, 0.18406398594379425, 0.09672915190458298, 0.3545641303062439, 0.23840880393981934, 0.08152862638235092, 0.19002404808998108], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_034b3071a2dd9cdd40d97b86df36fc31(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3136, 64], dtype='float16'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_aa219f413293acee1079f66bab1df3f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_034b3071a2dd9cdd40d97b86df36fc31
    def get_inputs(self):
        return [
            paddle.uniform([1, 3136, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fb2c253744cb9ed8d9f5808755f38927(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.18366307020187378, 0.12426714599132538, 0.4972164034843445, 0.062458768486976624, 0.4127230644226074, 0.20901690423488617, 0.0540064238011837, 0.4543137550354004, 0.41377779841423035, 0.13937672972679138, 0.2189370095729828, 0.2536654770374298, 0.04694389924407005, 0.25529488921165466, 0.052069131284952164, 0.3095897436141968, 0.3341934382915497, 0.23582299053668976, 0.1625525802373886, 0.1395953744649887, 0.4057944118976593, 0.41055068373680115, 0.35570409893989563, 0.35884156823158264], dtype='float32').reshape([24]),
            paddle.to_tensor([0.02742994949221611, 0.31682878732681274, 0.49627548456192017, 0.06683100759983063, 0.15778422355651855, 0.04458969458937645, 0.49738025665283203, 0.4496666491031647, 0.4337845742702484, 0.1711878627538681, 0.2398301362991333, 0.042817216366529465, 0.37853601574897766, 0.45221853256225586, 0.026275746524333954, 0.0619528666138649, 0.4249951243400574, 0.06234719231724739, 0.031480420380830765, 0.24545733630657196, 0.46106672286987305, 0.4392014443874359, 0.20094449818134308, 0.3954869210720062], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_276d3cd72d5c042b2517c757a858e517(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.29579704999923706, 0.3934045135974884, 0.2071964293718338, 0.03914392367005348, 0.3086318373680115, 0.2769215404987335, 0.10099522024393082, 0.005522829480469227, 0.3575942814350128, 0.1584794819355011, 0.3809984624385834, 0.471023291349411, 0.37874114513397217, 0.20067955553531647, 0.11812857538461685, 0.46527349948883057, 0.08282085508108139, 0.3970787227153778, 0.36169201135635376, 0.24797359108924866, 0.47906413674354553, 0.2564951479434967, 0.20490019023418427, 0.09064430743455887], dtype='float32').reshape([24]),
            paddle.to_tensor([0.11117620766162872, 0.10596942901611328, 0.20791016519069672, 0.038244303315877914, 0.4680157005786896, 0.3131575584411621, 0.2714845836162567, 0.10299527645111084, 0.48713934421539307, 0.4871602952480316, 0.28845202922821045, 0.25746169686317444, 0.14901545643806458, 0.19631822407245636, 0.2599518597126007, 0.44552749395370483, 0.4309538006782532, 0.19098258018493652, 0.3939175605773926, 0.4217641055583954, 0.45723047852516174, 0.10886441916227341, 0.23225022852420807, 0.22712258994579315], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_96e93aaec870fba9a738681ae51f25dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.014243364334106445, 0.2794649004936218, 0.32717716693878174, 0.47490912675857544, 0.22570741176605225, 0.16042979061603546, 0.37896236777305603, 0.030417995527386665, 0.4506978988647461, 0.11219741404056549, 0.20871514081954956, 0.32348334789276123, 0.15902097523212433, 0.04215121269226074, 0.33045271039009094, 0.0210556723177433, 0.49260541796684265, 0.34995031356811523, 0.09651444107294083, 0.13955333828926086, 0.02192113548517227, 0.16689813137054443, 0.48275765776634216, 0.11836960166692734], dtype='float32').reshape([24]),
            paddle.to_tensor([0.0507824569940567, 0.486506849527359, 0.46001583337783813, 0.1119002103805542, 0.2957666516304016, 0.34965643286705017, 0.33913567662239075, 0.21292796730995178, 0.40829792618751526, 0.2172970473766327, 0.3668845295906067, 0.07309071719646454, 0.13398326933383942, 0.02183481864631176, 0.0208249744027853, 0.0062780617736279964, 0.22229428589344025, 0.10834936797618866, 0.41508275270462036, 0.056368354707956314, 0.03811442479491234, 0.001178551698103547, 0.19034916162490845, 0.04245162382721901], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_4a03124c9258ab8dbad9b09aac642661(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9b9a79471f2248e0c5c2599b679c86c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a03124c9258ab8dbad9b09aac642661
    def get_inputs(self):
        return [
            paddle.uniform([4, 64, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f011bb45c016fa48c904fada82bf15db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.16475403308868408, 0.4828357398509979, 0.4258727729320526, 0.22399568557739258, 0.14013613760471344, 0.26562273502349854, 0.4289284348487854, 0.4146000146865845, 0.1591757982969284, 0.36893364787101746, 0.18424677848815918, 0.40686145424842834, 0.2382490634918213, 0.20175544917583466, 0.12711192667484283, 0.389931857585907, 0.06601797044277191, 0.42960959672927856, 0.3454381823539734, 0.0585675872862339, 0.44052815437316895, 0.37792930006980896, 0.1036595031619072, 0.27953049540519714], dtype='float32').reshape([24]),
            paddle.to_tensor([0.05428473278880119, 0.42193880677223206, 0.04669639840722084, 0.2255106270313263, 0.1590641438961029, 0.42216768860816956, 0.43623289465904236, 0.006709329783916473, 0.009879864752292633, 0.4408259391784668, 0.03323203697800636, 0.050958674401044846, 0.18800625205039978, 0.43866127729415894, 0.126575767993927, 0.44613292813301086, 0.4748630225658417, 0.2567873001098633, 0.21956348419189453, 0.46179062128067017, 0.13494503498077393, 0.11803732067346573, 0.0688214972615242, 0.06921400874853134], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b22b8b12e814910146f56dc7f1f253ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.25710543990135193, 0.3526293933391571, 0.4140579402446747, 0.4722927212715149, 0.2351761758327484, 0.37296032905578613, 0.0974608063697815, 0.19325564801692963, 0.011694647371768951, 0.16122131049633026, 0.04489336907863617, 0.1953994631767273, 0.23894086480140686, 0.1467740833759308, 0.15643905103206635, 0.010504195466637611, 0.17769835889339447, 0.0664338693022728, 0.36663544178009033, 0.03689739108085632, 0.3257657587528229, 0.22407668828964233, 0.41362735629081726, 0.22624360024929047], dtype='float32').reshape([24]),
            paddle.to_tensor([0.3817916512489319, 0.3329784572124481, 0.4374532997608185, 0.4319269061088562, 0.19798001646995544, 0.10686233639717102, 0.16284531354904175, 0.2656308710575104, 0.1156698539853096, 0.22367294132709503, 0.3040522634983063, 0.48494473099708557, 0.48920366168022156, 0.33050382137298584, 0.11264272034168243, 0.23953868448734283, 0.4199221432209015, 0.42665019631385803, 0.21511241793632507, 0.4858913719654083, 0.030990809202194214, 0.07033313065767288, 0.07851997762918472, 0.18951284885406494], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3fe8baa806348ddc17b9bdff27e17c7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.13936637341976166, 0.44048917293548584, 0.34544068574905396, 0.226340189576149, 0.11526307463645935, 0.30719825625419617, 0.2041853368282318, 0.290505588054657, 0.2068309783935547, 0.1575533002614975, 0.37899428606033325, 0.40094995498657227, 0.1259523332118988, 0.37002459168434143, 0.3306317925453186, 0.22001951932907104, 0.013940393924713135, 0.3108084499835968, 0.2097664624452591, 0.10541463643312454, 0.3320286273956299, 0.1915835440158844, 0.1417192965745926, 0.06755843013525009], dtype='float32').reshape([24]),
            paddle.to_tensor([0.1867671012878418, 0.04889613389968872, 0.47734805941581726, 0.1333237588405609, 0.07605227828025818, 0.4606616199016571, 0.4870370626449585, 0.06249965727329254, 0.4981561303138733, 0.4332980513572693, 0.11396840959787369, 0.34568482637405396, 0.15402203798294067, 0.2825644612312317, 0.3632968068122864, 0.36311984062194824, 0.13750992715358734, 0.2815728187561035, 0.39703822135925293, 0.08063396066427231, 0.13278478384017944, 0.11740539222955704, 0.3181067705154419, 0.1884966939687729], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_63c9116429cf16d6e20f420c1d4ccc9b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 240], dtype='float16'),
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            paddle.static.InputSpec(shape=[240], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b21e023a353c56a15175677d6015ce37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63c9116429cf16d6e20f420c1d4ccc9b
    def get_inputs(self):
        return [
            paddle.uniform([4, 16, 240], dtype='float16', min=0, max=0.5),
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ed582a184e8d55ff72002ee6f729d436(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1938668042421341, 0.07876656949520111, 0.09433778375387192, 0.4779868423938751, 0.34814438223838806, 0.4469517469406128, 0.43437811732292175, 0.159662127494812, 0.18601472675800323, 0.10298773646354675, 0.4830115735530853, 0.13508522510528564, 0.068057581782341, 0.23629240691661835, 0.0015584391076117754, 0.13886502385139465, 0.16179491579532623, 0.3291814625263214, 0.2886349856853485, 0.02491985447704792, 0.28625237941741943, 0.24895773828029633, 0.17609760165214539, 0.06893787533044815], dtype='float32').reshape([24]),
            paddle.to_tensor([0.049390219151973724, 0.2170150727033615, 0.2732374966144562, 0.4153362810611725, 0.06235528737306595, 0.20021355152130127, 0.18336232006549835, 0.06839175522327423, 0.2917318642139435, 0.3795059621334076, 0.10806344449520111, 0.16893772780895233, 0.03804459422826767, 0.05009950324892998, 0.2471439093351364, 0.4515213072299957, 0.03288698568940163, 0.4547441005706787, 0.23313893377780914, 0.12240840494632721, 0.22567322850227356, 0.03833785653114319, 0.37192702293395996, 0.20805618166923523], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_88a9245c5f4bdb9919540edc0f87bec7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 240], dtype='float32'),
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            paddle.static.InputSpec(shape=[240], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7e27956e17dc9690f4c1f87fca4dfa06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88a9245c5f4bdb9919540edc0f87bec7
    def get_inputs(self):
        return [
            paddle.uniform([4, 16, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e9cfbfa4e1474e8b1bcee2d9c59c7799(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.18731826543807983, 0.4828099310398102, 0.25584864616394043, 0.01906595006585121, 0.20454615354537964, 0.07630125433206558, 0.004274432547390461, 0.4532151222229004, 0.19866101443767548, 0.20867544412612915, 0.12019315361976624, 0.22176975011825562, 0.2413586527109146, 0.07520231604576111, 0.35733622312545776, 0.379110723733902, 0.41227513551712036, 0.1334734708070755, 0.36011257767677307, 0.39803704619407654, 0.10016573220491409, 0.08654404431581497, 0.06558667123317719, 0.20992836356163025], dtype='float32').reshape([24]),
            paddle.to_tensor([0.3008992075920105, 0.06400731950998306, 0.16011938452720642, 0.2368287593126297, 0.48954954743385315, 0.07106383889913559, 0.018838316202163696, 0.3443790674209595, 0.4682483673095703, 0.12135474383831024, 0.06300223618745804, 0.4523089826107025, 0.2626247704029083, 0.04270777851343155, 0.13154040277004242, 0.010985239408910275, 0.31751522421836853, 0.2806195914745331, 0.4354870021343231, 0.45831331610679626, 0.1530156433582306, 0.3335305154323578, 0.10656460374593735, 0.18355105817317963], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_adba015e4075934edea2be396e519460(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c96e2de8d6f3122368aef5de69773acf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adba015e4075934edea2be396e519460
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_fb06dea1c2e2abee1dc3b14d3edf6867(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 32], dtype='float16'),
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            paddle.static.InputSpec(shape=[32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_47fd55605b5e8a82203ce3b1c2c6e811(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb06dea1c2e2abee1dc3b14d3edf6867
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 32], dtype='float16', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_117b1f7c2890ad6e03295b828e28b15e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.010870775207877159, 0.03675542026758194, 0.24669773876667023, 0.05390876159071922, 0.316701203584671, 0.17424993216991425, 0.03908360376954079, 0.43361032009124756, 0.4712028503417969, 0.3705647587776184, 0.07935892790555954, 0.0350777767598629, 0.41310372948646545, 0.3556474447250366, 0.02611469477415085, 0.11116866767406464, 0.07265156507492065, 0.3031384348869324, 0.291408509016037, 0.49377259612083435, 0.17582684755325317, 0.07363982498645782, 0.09489616751670837, 0.35881510376930237], dtype='float32').reshape([24]),
            paddle.to_tensor([0.22165365517139435, 0.22210361063480377, 0.25007158517837524, 0.04683087021112442, 0.0700167864561081, 0.27537307143211365, 0.4492897391319275, 0.4553984999656677, 0.17673273384571075, 0.4537613093852997, 0.47669342160224915, 0.448720246553421, 0.4232642650604248, 0.0006090207607485354, 0.2675057649612427, 0.46675780415534973, 0.38979390263557434, 0.3350133001804352, 0.22259560227394104, 0.003308809595182538, 0.03921572491526604, 0.4619683623313904, 0.48399123549461365, 0.4778873026371002], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_5077cb31a25157d88979de23e00f4793(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 196, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4d10de05e75bd5b25a8a0c8e3fdcdc99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5077cb31a25157d88979de23e00f4793
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_655616a8789ba66b34a6c297ec8ce0d8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 512], dtype='float16'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_31cd1cdac5f2a17d3b48119f9dd72f30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655616a8789ba66b34a6c297ec8ce0d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a9a160bb1d6c81b6b869c69aba44c12c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.33882394433021545, 0.3488764464855194, 0.03562818467617035, 0.3791046738624573, 0.3421655297279358, 0.13304094970226288, 0.18232062458992004, 0.14956778287887573, 0.052485957741737366, 0.3806203305721283, 0.2293543964624405, 0.18278725445270538, 0.20754069089889526, 0.2983909249305725, 0.48287203907966614, 0.24650907516479492, 0.19480866193771362, 0.26233288645744324, 0.4814402759075165, 0.12110041826963425, 0.0721556767821312, 0.35488057136535645, 0.4377501904964447, 0.05875638872385025], dtype='float32').reshape([24]),
            paddle.to_tensor([0.3843730688095093, 0.1805105358362198, 0.40081948041915894, 0.1916782408952713, 0.28682342171669006, 0.44120410084724426, 0.417477011680603, 0.20276883244514465, 0.35341861844062805, 0.3918842673301697, 0.19456268846988678, 0.15923380851745605, 0.06132087856531143, 0.30830180644989014, 0.1771114468574524, 0.2213728427886963, 0.3380354642868042, 0.01230168528854847, 0.45821240544319153, 0.22713106870651245, 0.3093269169330597, 0.3850228786468506, 0.11108630895614624, 0.022809235379099846], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_2ef84b6b4ad597fb38bc99418e87778d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 192], dtype='float16'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1a84a210ae96f6729f8eadec0adfb1cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ef84b6b4ad597fb38bc99418e87778d
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 192], dtype='float16', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7197cf213f670e2a0d95ce7b9446cbc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.07532916218042374, 0.1972924917936325, 0.4818250238895416, 0.34716910123825073, 0.1353168487548828, 0.12077910453081131, 0.4096485674381256, 0.42836514115333557, 0.14646293222904205, 0.21278241276741028, 0.46058011054992676, 0.2457815706729889, 0.3760378956794739, 0.046136524528265, 0.22805488109588623, 0.1155649796128273, 0.32413366436958313, 0.49832016229629517, 0.42351728677749634, 0.08922522515058517, 0.23585352301597595, 0.08159761875867844, 0.33681657910346985, 0.3886728286743164], dtype='float32').reshape([24]),
            paddle.to_tensor([0.186040461063385, 0.3528192639350891, 0.09311974048614502, 0.22022080421447754, 0.03183605521917343, 0.23108990490436554, 0.05479612946510315, 0.0018270742148160934, 0.2949790060520172, 0.3881951570510864, 0.11235623061656952, 0.3297078013420105, 0.4999403953552246, 0.18454866111278534, 0.4581121802330017, 0.43436485528945923, 0.07245106250047684, 0.09357249736785889, 0.30169519782066345, 0.2772696912288666, 0.3879294991493225, 0.00434198509901762, 0.10458900034427643, 0.4332750141620636], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8185cb38f17ee868f7421236d3a4ad2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.15723566710948944, 0.2434181272983551, 0.012888051569461823, 0.0034653411712497473, 0.4338573217391968, 0.10937909781932831, 0.46160873770713806, 0.22581854462623596, 0.32471129298210144, 0.40940603613853455, 0.3611995279788971, 0.073187455534935, 0.025003740563988686, 0.385524719953537, 0.09263664484024048, 0.03777698799967766, 0.15941482782363892, 0.017981860786676407, 0.28543469309806824, 0.27014341950416565, 0.3705376386642456, 0.01776932366192341, 0.26192259788513184, 0.22764934599399567], dtype='float32').reshape([24]),
            paddle.to_tensor([0.37588319182395935, 0.4547136425971985, 0.1828862726688385, 0.1647234857082367, 0.04327135160565376, 0.018667694181203842, 0.2758021652698517, 0.4080827832221985, 0.10054291039705276, 0.4233485758304596, 0.3048818111419678, 0.1888388842344284, 0.45933932065963745, 0.35873204469680786, 0.45759496092796326, 0.45182090997695923, 0.17913800477981567, 0.12784822285175323, 0.2736762464046478, 0.4633031189441681, 0.150021493434906, 0.3244920074939728, 0.3048169016838074, 0.34645697474479675], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ded8a2e4056fa49677663f338349f974(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.29522278904914856, 0.0950869619846344, 0.05496954545378685, 0.46835246682167053, 0.23117777705192566, 0.4735710918903351, 0.4322499930858612, 0.3404623568058014, 0.3642272353172302, 0.04208987578749657, 0.12262923270463943, 0.078940249979496, 0.1275501549243927, 0.4027003347873688, 0.0009189197444356978, 0.4557492434978485, 0.20845770835876465, 0.238201305270195, 0.0010032330173999071, 0.19651584327220917, 0.15796522796154022, 0.3953298032283783, 0.006185709033161402, 0.13807445764541626], dtype='float32').reshape([24]),
            paddle.to_tensor([0.38487422466278076, 0.030802318826317787, 0.46444231271743774, 0.22993767261505127, 0.15590202808380127, 0.07117673009634018, 0.0986221432685852, 0.3898000121116638, 0.21487857401371002, 0.10428028553724289, 0.34426283836364746, 0.1759105771780014, 0.0740404799580574, 0.4490010142326355, 0.13167668879032135, 0.06733668595552444, 0.14908458292484283, 0.058156583458185196, 0.330643892288208, 0.4362650513648987, 0.23603256046772003, 0.22355514764785767, 0.37005114555358887, 0.3322056829929352], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_0a38eac31623f166b869c7d0cf3df78d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3136, 32], dtype='float16'),
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            paddle.static.InputSpec(shape=[32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_235673f38fb88d0745508d8e4f3c4591(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a38eac31623f166b869c7d0cf3df78d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3136, 32], dtype='float16', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f294029a6978e7f3297a66b66d6179c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1682705283164978, 0.3565743863582611, 0.1253589242696762, 0.16418196260929108, 0.3286559581756592, 0.48541876673698425, 0.2749499976634979, 0.4234136939048767, 0.08529394119977951, 0.13429801166057587, 0.07868754863739014, 0.00985343474894762, 0.06885045766830444, 0.35013991594314575, 0.22062887251377106, 0.14742974936962128, 0.39567428827285767, 0.3646829426288605, 0.24926216900348663, 0.2954123318195343, 0.01851140707731247, 0.020235564559698105, 0.179886594414711, 0.06986070424318314], dtype='float32').reshape([24]),
            paddle.to_tensor([0.22930927574634552, 0.36494237184524536, 0.18423578143119812, 0.46554070711135864, 0.1347815990447998, 0.20376929640769958, 0.25552263855934143, 0.2063751220703125, 0.40140387415885925, 0.4490581154823303, 0.39951521158218384, 0.0205830130726099, 0.2740081250667572, 0.48853519558906555, 0.2794617712497711, 0.15489323437213898, 0.44764864444732666, 0.02543565444648266, 0.002081912476569414, 0.2824811637401581, 0.1383700966835022, 0.12855780124664307, 0.056254513561725616, 0.35080331563949585], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e44a7be4fe0f990a687e9700e7e16e10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1726970374584198, 0.16181598603725433, 0.10104984790086746, 0.3545682728290558, 0.2902470827102661, 0.23746728897094727, 0.30638277530670166, 0.029487179592251778, 0.07071344554424286, 0.489443302154541, 0.4145127832889557, 0.13114866614341736, 0.007585819344967604, 0.006573534104973078, 0.29442575573921204, 0.21814538538455963, 0.0454707071185112, 0.459863543510437, 0.16221505403518677, 0.4264506697654724, 0.08130539208650589, 0.27876222133636475, 0.22132013738155365, 0.3513026833534241], dtype='float32').reshape([24]),
            paddle.to_tensor([0.29074880480766296, 0.24003681540489197, 0.0007325326441787183, 0.4447775185108185, 0.23070864379405975, 0.10364915430545807, 0.4267929196357727, 0.38438647985458374, 0.15183788537979126, 0.41052234172821045, 0.3081268072128296, 0.07744412869215012, 0.09051049500703812, 0.031048299744725227, 0.34466007351875305, 0.08504622429609299, 0.005109978839755058, 0.4140075445175171, 0.22425530850887299, 0.1307162195444107, 0.24897487461566925, 0.1401134878396988, 0.25636357069015503, 0.017722351476550102], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_533cb5d8a1692b32260c81677787763f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.33938372135162354, 0.40610218048095703, 0.30375155806541443, 0.2319847196340561, 0.11537652462720871, 0.432975709438324, 0.22722597420215607, 0.2257462590932846, 0.41630417108535767, 0.443502277135849, 0.37145131826400757, 0.31059888005256653, 0.42944005131721497, 0.2317645400762558, 0.3220788240432739, 0.3324247896671295, 0.32443344593048096, 0.2245652675628662, 0.07454076409339905, 0.047343313694000244, 0.22435623407363892, 0.06307432055473328, 0.07455383241176605, 0.24440547823905945], dtype='float32').reshape([24]),
            paddle.to_tensor([0.2475200593471527, 0.3294788599014282, 0.07380232214927673, 0.4451567232608795, 0.14242061972618103, 0.27744394540786743, 0.4061700403690338, 0.257845401763916, 0.05833157151937485, 0.21847888827323914, 0.46438068151474, 0.16107966005802155, 0.08478806167840958, 0.3496074974536896, 0.004090358968824148, 0.3098233640193939, 0.35273170471191406, 0.3347354829311371, 0.19040662050247192, 0.3860618472099304, 0.2835719883441925, 0.2049616575241089, 0.437127947807312, 0.11209271848201752], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3a56b062deb6b9ff8ff530ce083494ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4525972604751587, 0.10570936650037766, 0.19949999451637268, 0.4838881194591522, 0.2377445548772812, 0.17726634442806244, 0.24631813168525696, 0.30529892444610596, 0.24530333280563354, 0.2580033838748932, 0.2084118276834488, 0.4892955720424652, 0.04473545029759407, 0.27066031098365784, 0.4945635199546814, 0.4525549113750458, 0.18380023539066315, 0.14176037907600403, 0.21209178864955902, 0.1922387033700943, 0.3587110936641693, 0.009046556428074837, 0.21292348206043243, 0.06784713268280029], dtype='float32').reshape([24]),
            paddle.to_tensor([0.036773622035980225, 0.12827908992767334, 0.4302771985530853, 0.35752516984939575, 0.2255377471446991, 0.3609851002693176, 0.37519529461860657, 0.17342035472393036, 0.2122988998889923, 0.34364548325538635, 0.41927674412727356, 0.038604021072387695, 0.42805013060569763, 0.39877498149871826, 0.08990813791751862, 0.38422563672065735, 0.08769549429416656, 0.4307102859020233, 0.08969058096408844, 0.3177984952926636, 0.31290921568870544, 0.28953176736831665, 0.12662701308727264, 0.30445700883865356], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6286281eef6f5d7aedd49fcaaac3deca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.18257200717926025, 0.43181341886520386, 0.30533620715141296, 0.14209537208080292, 0.29281044006347656, 0.040524162352085114, 0.339503675699234, 0.12490895390510559, 0.002219052752479911, 0.24146921932697296, 0.1458093523979187, 0.4019382894039154, 0.26807624101638794, 0.3998703956604004, 0.350229948759079, 0.4934419095516205, 0.08212210237979889, 0.35577523708343506, 0.23343515396118164, 0.2782115340232849, 0.36634939908981323, 0.07314932346343994, 0.053259674459695816, 0.32733631134033203], dtype='float32').reshape([24]),
            paddle.to_tensor([0.13197894394397736, 0.13877402245998383, 0.18933744728565216, 0.4786091148853302, 0.24661923944950104, 0.021998360753059387, 0.4621764123439789, 0.15356013178825378, 0.19081975519657135, 0.4565843641757965, 0.4964907467365265, 0.45384037494659424, 0.14349815249443054, 0.368434876203537, 0.13210786879062653, 0.4741015136241913, 0.38814622163772583, 0.09981435537338257, 0.08217289298772812, 0.05007820948958397, 0.0030590062960982323, 0.04989596828818321, 0.3537556529045105, 0.37192681431770325], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_72e46c64704abb74b1cba7fd943c1f73(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 144], dtype='float32'),
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            paddle.static.InputSpec(shape=[144], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ae05866749ee1d417b14a639eb81fb77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72e46c64704abb74b1cba7fd943c1f73
    def get_inputs(self):
        return [
            paddle.uniform([4, 256, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5e8bed061b1bdcd9637e885ec972a78d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3968702554702759, 0.4320816397666931, 0.3889198303222656, 0.30037838220596313, 0.11799021810293198, 0.4648837745189667, 0.03690313920378685, 0.05396295338869095, 0.46167248487472534, 0.28702905774116516, 0.4455452859401703, 0.46462130546569824, 0.46802258491516113, 0.1937229484319687, 0.3695943355560303, 0.0727655217051506, 0.0497843399643898, 0.3123270273208618, 0.15348221361637115, 0.05088169872760773, 0.30928343534469604, 0.14680342376232147, 0.48877212405204773, 0.037296999245882034], dtype='float32').reshape([24]),
            paddle.to_tensor([0.2778010964393616, 0.27052295207977295, 0.19602876901626587, 0.1684628129005432, 0.4654140770435333, 0.3379822075366974, 0.2950296401977539, 0.17189368605613708, 0.005182778928428888, 0.18551237881183624, 0.05109640210866928, 0.4282330274581909, 0.21332058310508728, 0.14822576940059662, 0.0646529421210289, 0.3894656300544739, 0.313083678483963, 0.03633873537182808, 0.410410612821579, 0.1478414684534073, 0.3513844311237335, 0.06024102866649628, 0.3374421000480652, 0.1095222607254982], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b49cdfee21a57b53f75bb38e3342006c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.30301332473754883, 0.4493446946144104, 0.4292582869529724, 0.3049544095993042, 0.08340997993946075, 0.3223626911640167, 0.1273801475763321, 0.17693401873111725, 0.04197666049003601, 0.49660390615463257, 0.4233519732952118, 0.37309715151786804, 0.31765690445899963, 0.14948458969593048, 0.004608043469488621, 0.2522963285446167, 0.21284528076648712, 0.05863044038414955, 0.07177890837192535, 0.32406264543533325, 0.2306806594133377, 0.10358590632677078, 0.1974070966243744, 0.32947397232055664], dtype='float32').reshape([24]),
            paddle.to_tensor([0.18616163730621338, 0.28802230954170227, 0.21353426575660706, 0.3143126964569092, 0.4350679814815521, 0.0990566536784172, 0.35628223419189453, 0.0036044223234057426, 0.12603428959846497, 0.2742798924446106, 0.49496158957481384, 0.23257498443126678, 0.44189727306365967, 0.305686891078949, 0.4327239990234375, 0.054918453097343445, 0.006196706555783749, 0.253266304731369, 0.32935842871665955, 0.025479452684521675, 0.24025894701480865, 0.3470144271850586, 0.33367231488227844, 0.440521240234375], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_727d751a20cd38b7c6e1f9893adad5ab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_19f1610724e23fcfaf179c4660e639a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_727d751a20cd38b7c6e1f9893adad5ab
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_9e095bc50434233938a51734fc6647af(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 576, 1024], dtype='float16'),
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d746ef79142ade9d6b68f5ae5d9f7a35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e095bc50434233938a51734fc6647af
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1024], dtype='float16', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_22ac592ec353803653cc8d60859390b6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 96], dtype='float16'),
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            paddle.static.InputSpec(shape=[96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f02a41e7cc1b5ed1381771a556697d9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22ac592ec353803653cc8d60859390b6
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 96], dtype='float16', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6c527fa7e0d2d06c09edae51a9124eb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.06041004881262779, 0.3167206943035126, 0.05374542996287346, 0.36268866062164307, 0.19197198748588562, 0.4946305751800537, 0.12886005640029907, 0.14391383528709412, 0.16145367920398712, 0.3001125752925873, 0.18410956859588623, 0.18207840621471405, 0.4029044806957245, 0.06203930452466011, 0.144857719540596, 0.2986207604408264, 0.2609812319278717, 0.32699164748191833, 0.18422389030456543, 0.48522546887397766, 0.15709753334522247, 0.37659984827041626, 0.18521958589553833, 0.14869895577430725], dtype='float32').reshape([24]),
            paddle.to_tensor([0.31494954228401184, 0.17287783324718475, 0.16401103138923645, 0.4613477289676666, 0.04905761033296585, 0.464922696352005, 0.11221131682395935, 0.2821165919303894, 0.29332780838012695, 0.019930627197027206, 0.2986953556537628, 0.35327333211898804, 0.3241746723651886, 0.2458716630935669, 0.4133024513721466, 0.07294643670320511, 0.20899417996406555, 0.20976218581199646, 0.1856888234615326, 0.425304114818573, 0.1442001610994339, 0.18219241499900818, 0.18387548625469208, 0.4614901840686798], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_06be4ba2a5327b5645f1e6b349610541(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.33545830845832825, 0.37854981422424316, 0.1908382773399353, 0.05427590012550354, 0.1394834965467453, 0.14096412062644958, 0.13863807916641235, 0.39346635341644287, 0.19579078257083893, 0.010661977343261242, 0.4447462558746338, 0.3711831569671631, 0.07698850333690643, 0.09032592922449112, 0.38182857632637024, 0.35339832305908203, 0.15278765559196472, 0.18944057822227478, 0.2009609043598175, 0.08842156827449799, 0.1725178062915802, 0.3571804165840149, 0.030272960662841797, 0.48736661672592163], dtype='float32').reshape([24]),
            paddle.to_tensor([0.285391241312027, 0.12025842815637589, 0.08677830547094345, 0.04854026809334755, 0.026385126635432243, 0.4224960207939148, 0.32764673233032227, 0.4977532923221588, 0.4031577408313751, 0.37871429324150085, 0.05479966849088669, 0.1505054086446762, 0.003360736183822155, 0.11643196642398834, 0.28942087292671204, 0.10076363384723663, 0.21567189693450928, 0.1498848795890808, 0.15887169539928436, 0.0488554947078228, 0.24201691150665283, 0.34989067912101746, 0.20457899570465088, 0.49740856885910034], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()


if __name__ == '__main__':
    unittest.main()