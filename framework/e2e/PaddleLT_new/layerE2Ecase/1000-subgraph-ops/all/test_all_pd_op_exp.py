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
class PrimitiveOp_5aa6d3e282b1ed0a8555da8b30502da5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.exp(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f1112e11dbe869c419ccd17ccf8c76fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aa6d3e282b1ed0a8555da8b30502da5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.08470913767814636]], [[0.24686630070209503]], [[0.33088260889053345]], [[0.011963418684899807]], [[0.3026564121246338]], [[0.48026254773139954]]], dtype='float32').reshape([6, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_7bd4df376b51a12811146f1ebbd12b33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aa6d3e282b1ed0a8555da8b30502da5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.3118162155151367]], [[0.4953351318836212]], [[0.017668241634964943]], [[0.26220181584358215]], [[0.12516820430755615]], [[0.3981139659881592]]], dtype='float32').reshape([6, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_bb79c97fa4bd42b50cb8ae0d7bede263(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aa6d3e282b1ed0a8555da8b30502da5
    def get_inputs(self):
        return [
            paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_6903f5d8e144b8af83135f61921a9616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aa6d3e282b1ed0a8555da8b30502da5
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_963bb3becb286fb5590ad1b5774ecafe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aa6d3e282b1ed0a8555da8b30502da5
    def get_inputs(self):
        return [
            paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_25db75b1125d3813894204a4020f47da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aa6d3e282b1ed0a8555da8b30502da5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.07869533449411392]], [[0.4570064842700958]], [[0.32425031065940857]], [[0.26880863308906555]], [[0.07264065742492676]], [[0.44993671774864197]]], dtype='float32').reshape([6, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_06f70542e26dc1b38871ec7a6860bc1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aa6d3e282b1ed0a8555da8b30502da5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[0.30910688638687134]], [[0.3038202226161957]], [[0.050439853221178055]], [[0.46539199352264404]], [[0.0930788591504097]], [[0.21133345365524292]]], dtype='float32').reshape([6, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_5daf61cbbb99625aae77c69c7336f1c2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.exp(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f695d57a89147f6c228318455847bc8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5daf61cbbb99625aae77c69c7336f1c2
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.21487902104854584], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_6b3046be120f28f594d2c3a42d13d459(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5daf61cbbb99625aae77c69c7336f1c2
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.002732213120907545], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_0a21e0fae1c1b583ced1fbd8851c57bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aa6d3e282b1ed0a8555da8b30502da5
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_72c565b6fb0c9a1f3afc92fa56af560f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5daf61cbbb99625aae77c69c7336f1c2
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.3639618754386902], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_bb4aafed779e84ec7248fb23f756d58c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5daf61cbbb99625aae77c69c7336f1c2
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.05983737111091614], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_c1a1fcf7b8b1e1fe3e847f9679c4d15d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5daf61cbbb99625aae77c69c7336f1c2
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.49071863293647766], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_11fef5dfc4a2814681336ee27ca56c6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5daf61cbbb99625aae77c69c7336f1c2
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.014714278280735016], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_7e6ce8658a95f40a5a674a9e62578a8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5daf61cbbb99625aae77c69c7336f1c2
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.11325055360794067], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_8bc55b848c30ecc37097c289eba9abf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5daf61cbbb99625aae77c69c7336f1c2
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.25846362113952637], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_ec9c876f7e08d02d08e2e081fffe8745(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5daf61cbbb99625aae77c69c7336f1c2
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.3633100092411041], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_b1995b06b49aea8e89efc429b121df4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5daf61cbbb99625aae77c69c7336f1c2
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.23192593455314636], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_8b9b4eec9c064511df99955b570657af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5daf61cbbb99625aae77c69c7336f1c2
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.389018714427948], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_01ef1f16b56edd25192367e2439aef3d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.exp(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_31080772e5ff750121087a01fee3a373(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01ef1f16b56edd25192367e2439aef3d
    def get_inputs(self):
        return [
            paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_1b10e68a782c531aa56aa5b09ad1f3cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aa6d3e282b1ed0a8555da8b30502da5
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_c17abc196a3b4590a37800e65746b974(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5aa6d3e282b1ed0a8555da8b30502da5
    def get_inputs(self):
        return [
            paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_1f496528a2df5c3ffd24db398ee0341c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.exp(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_06df2983adac3145128e4a7570df0b9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f496528a2df5c3ffd24db398ee0341c
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.21487902104854584], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_f2bc40930272d4bcd696f1bba97533e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f496528a2df5c3ffd24db398ee0341c
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.002732213120907545], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_9735ca8bd2b29b856d5ee8cde9e717a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f496528a2df5c3ffd24db398ee0341c
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.3639618754386902], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_7523eda09fe7ae53138281041b461a91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f496528a2df5c3ffd24db398ee0341c
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.05983737111091614], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_9bf4d53d60903b35a04eebe6ee98ba10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f496528a2df5c3ffd24db398ee0341c
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.49071863293647766], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_1f677e6c17d558c86e775986dca7f71a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f496528a2df5c3ffd24db398ee0341c
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.014714278280735016], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_4bcae8fad9ae1a015b5275f5ce3a64cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f496528a2df5c3ffd24db398ee0341c
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.11325055360794067], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_6e43db435387d609589fbbbba9fca02e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f496528a2df5c3ffd24db398ee0341c
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.25846362113952637], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_ca840680660114e8f870a8d03a697806(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f496528a2df5c3ffd24db398ee0341c
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.3633100092411041], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_05fab23b0685d1eec23d74338b520399(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f496528a2df5c3ffd24db398ee0341c
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.23192593455314636], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_61b5b8bd98e375eaf48af29fded22e7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f496528a2df5c3ffd24db398ee0341c
    def get_inputs(self):
        return [
            paddle.to_tensor([-0.389018714427948], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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