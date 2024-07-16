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
class PrimitiveOp_5bc7ea742aa14540ff5fdd6163035046(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.0
        input_2 = 1.0
        return paddle._C_ops.clip(input_0, min(input_1, input_2), max(input_1, input_2))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8edd6d85f2ae77ec74051e52ed061857(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bc7ea742aa14540ff5fdd6163035046
    def get_inputs(self):
        return [
            paddle.uniform([1, 72], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_90b4b86c315f3487028bd085b4a975c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bc7ea742aa14540ff5fdd6163035046
    def get_inputs(self):
        return [
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_a72f77f779d184e38083a31a3c24e068(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bc7ea742aa14540ff5fdd6163035046
    def get_inputs(self):
        return [
            paddle.uniform([1, 960], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_44eb06f2f3ea2bbfab36eb8c4e2d4e2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bc7ea742aa14540ff5fdd6163035046
    def get_inputs(self):
        return [
            paddle.uniform([1, 480], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_554e3ca10835228074ccd9075556a557(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = -10000000000.0
        input_2 = 4.135169982910156
        return paddle._C_ops.clip(input_0, min(input_1, input_2), max(input_1, input_2))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cf163450a6ea029a2b4e494756c25df8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_554e3ca10835228074ccd9075556a557
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

class PrimitiveOp_471e37956592534c29c856a2de770c75(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.0
        input_2 = 3.402820018375656e+38
        return paddle._C_ops.clip(input_0, min(input_1, input_2), max(input_1, input_2))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_efd6b51ad3d7f9a7e9ed22b91647f246(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_471e37956592534c29c856a2de770c75
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_b78244eaaaa83176ce591bfcc869cd83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bc7ea742aa14540ff5fdd6163035046
    def get_inputs(self):
        return [
            paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_daf31807cc90a5954cf0cb6d8379df95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_471e37956592534c29c856a2de770c75
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_e6b913bd82860298f0454022ba89f933(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bc7ea742aa14540ff5fdd6163035046
    def get_inputs(self):
        return [
            paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_731fc00cf6e6d14a62d3c3f69383be7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_471e37956592534c29c856a2de770c75
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_32fc00b5e65d31627ae4d539a11ef9a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_471e37956592534c29c856a2de770c75
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_c1cd956b1a807ab11726254239771d94(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1.0
        input_2 = 3.402820018375656e+38
        return paddle._C_ops.clip(input_0, min(input_1, input_2), max(input_1, input_2))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6e2acaa8ed69f6d58111bcc57175f84a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1cd956b1a807ab11726254239771d94
    def get_inputs(self):
        return [
            paddle.to_tensor(8732.0, dtype='float32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_620b7db6a9da4a0f140e56928ce820a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bc7ea742aa14540ff5fdd6163035046
    def get_inputs(self):
        return [
            paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_f05b0d11f6778630146b8b480f19706c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_471e37956592534c29c856a2de770c75
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_6703e09f1c24a52b0bfbf6984425c1a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bc7ea742aa14540ff5fdd6163035046
    def get_inputs(self):
        return [
            paddle.uniform([145, 240], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_bf98292c25ec99ff3fc8b9ff82c8ea93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_471e37956592534c29c856a2de770c75
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_f2d1c3e75db2168c27a74d4dd0fa5f6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_471e37956592534c29c856a2de770c75
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_f327252c899054a51870b7099a9d042c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bc7ea742aa14540ff5fdd6163035046
    def get_inputs(self):
        return [
            paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_18b8c7fbe0044b4ff3b5b2c375663cc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bc7ea742aa14540ff5fdd6163035046
    def get_inputs(self):
        return [
            paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_fabf2655118b286a4536724b7cff4c21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_471e37956592534c29c856a2de770c75
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_bb1fa392482923727e339b86e1855178(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.0
        input_2 = 3.402820018375656e+38
        return paddle._C_ops.clip(input_0, min(input_1, input_2), max(input_1, input_2))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_70ae5901a987c4acf58872e3812d89d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb1fa392482923727e339b86e1855178
    def get_inputs(self):
        return [
            paddle.uniform([1812, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_c46c934312dd803dbe9571cdfed8240a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.0
        input_2 = 15.989999771118164
        return paddle._C_ops.clip(input_0, min(input_1, input_2), max(input_1, input_2))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4adf17e6ad4fb6da060aceb955107b99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c46c934312dd803dbe9571cdfed8240a
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_700ae5c01a9f64a38273e1f9cfe8f311(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_471e37956592534c29c856a2de770c75
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_8d4301c455199d56f2c1761b55151c95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_471e37956592534c29c856a2de770c75
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_630cab4b5b56164293b4debaae9cfb6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bc7ea742aa14540ff5fdd6163035046
    def get_inputs(self):
        return [
            paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_b0da3012d88c46d5d21a89cbb7378323(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_471e37956592534c29c856a2de770c75
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_e50aa77b3ed35ba3d422ee32f650e58d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_471e37956592534c29c856a2de770c75
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_8ccd6c3b0e946d4e3b2edd838e6dc435(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb1fa392482923727e339b86e1855178
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.2358398288488388], [-0.3426547050476074], [-0.26866966485977173], [-0.27847227454185486], [-0.4832550585269928], [-0.2657580077648163], [-0.1454414278268814], [-0.3046749532222748], [-0.30635911226272583]], dtype='float32').reshape([9, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_b51972e4bf968753dc3c1fc6113ed969(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb1fa392482923727e339b86e1855178
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.14691856503486633], [-0.048890143632888794], [-0.16104330122470856], [-0.26051414012908936], [-0.3349742591381073], [-0.27397021651268005], [-0.009722471237182617], [-0.0020090043544769287], [-0.2285393327474594]], dtype='float32').reshape([9, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_2810173fcbbe4962003c65ff9d49ce65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bc7ea742aa14540ff5fdd6163035046
    def get_inputs(self):
        return [
            paddle.uniform([10, 240], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_d5ab047ab86dbe6be5260b30ef23b14b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_554e3ca10835228074ccd9075556a557
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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9736c8096214e4edc49fcf3ba8027c74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb1fa392482923727e339b86e1855178
    def get_inputs(self):
        return [
            paddle.uniform([5519, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_54dceedb87bdcf745d325f4126ee57ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c46c934312dd803dbe9571cdfed8240a
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3a1896d5ec36462db5b9a3b9e64616bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bc7ea742aa14540ff5fdd6163035046
    def get_inputs(self):
        return [
            paddle.uniform([10, 36], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_c3bea343cec9366e70bf74cfa787b3b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bc7ea742aa14540ff5fdd6163035046
    def get_inputs(self):
        return [
            paddle.uniform([10, 480], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_571ad7e63e6e621a3c35539744812c3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb1fa392482923727e339b86e1855178
    def get_inputs(self):
        return [
            paddle.uniform([1798, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_96eeebd42f55f8375e42344e3136e433(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = -2.0
        input_2 = 15.989999771118164
        return paddle._C_ops.clip(input_0, min(input_1, input_2), max(input_1, input_2))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5304c5b5626c767b3883b26833ede5f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96eeebd42f55f8375e42344e3136e433
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_99ee02b6c6c5e57a67363145fc4f963b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bc7ea742aa14540ff5fdd6163035046
    def get_inputs(self):
        return [
            paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_2ceb654c726a2295f077fc77222ce104(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.0
        input_2 = 1.0
        return paddle._C_ops.clip(input_0, min(input_1, input_2), max(input_1, input_2))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b7e12c0f201d4e868ff1dbbb2540066c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ceb654c726a2295f077fc77222ce104
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 100, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_4609921ece703d66fbca3a0d443b0f82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bc7ea742aa14540ff5fdd6163035046
    def get_inputs(self):
        return [
            paddle.uniform([171, 240], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_071c2c0cbf31e8791769e397559a1974(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bc7ea742aa14540ff5fdd6163035046
    def get_inputs(self):
        return [
            paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_a29ca8db008bebfe7ed2756dd7b12dff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb1fa392482923727e339b86e1855178
    def get_inputs(self):
        return [
            paddle.uniform([1574, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_05f47c1483a0afebd3a3c8a098b11ed5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c46c934312dd803dbe9571cdfed8240a
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9885febb12d87c9a64f75ad03e70e505(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bc7ea742aa14540ff5fdd6163035046
    def get_inputs(self):
        return [
            paddle.uniform([22, 240], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_1f660d78920ea62bbd59c1ba7fedba21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bc7ea742aa14540ff5fdd6163035046
    def get_inputs(self):
        return [
            paddle.uniform([22, 36], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_4263a9e609785335ad5f0d799ec4f6b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb1fa392482923727e339b86e1855178
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.266682893037796]], dtype='float32').reshape([1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_c98ee45006d681f6db7b9a28d9546824(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb1fa392482923727e339b86e1855178
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.43490374088287354]], dtype='float32').reshape([1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_f30cf4da8427f8ed8a38dab200f47dd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb1fa392482923727e339b86e1855178
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.15596023201942444], [-0.07294145226478577], [-0.2706569731235504], [-0.2249203324317932], [-0.2318911850452423], [-0.04057793319225311]], dtype='float32').reshape([6, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_41f4daba937345d0abc6cada464b8dca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb1fa392482923727e339b86e1855178
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.3357069790363312], [-0.11135042458772659], [0.34237468242645264], [-0.04790234565734863], [0.09837071597576141], [-0.3620920181274414]], dtype='float32').reshape([6, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_87bcb91e6a31e3df34e839e9f689bb72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bc7ea742aa14540ff5fdd6163035046
    def get_inputs(self):
        return [
            paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_e7f46885dc5adec09da7eb998b492ab5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bc7ea742aa14540ff5fdd6163035046
    def get_inputs(self):
        return [
            paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_992a1cd9103e47f0c9ae887e46d1e6f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb1fa392482923727e339b86e1855178
    def get_inputs(self):
        return [
            paddle.uniform([2055, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_e3de346ec5f32eb825fab5263e1b10d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c46c934312dd803dbe9571cdfed8240a
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_797c3625a6b7d4e0c5c49e053c1b92f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_471e37956592534c29c856a2de770c75
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_967d1a17ed2fa2000131288f283887d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb1fa392482923727e339b86e1855178
    def get_inputs(self):
        return [
            paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_068e4a7d8b26c908d9d5bce03614780e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c46c934312dd803dbe9571cdfed8240a
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1a41ee52aa588e0d41d776ffe0c7ea33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1cd956b1a807ab11726254239771d94
    def get_inputs(self):
        return [
            paddle.to_tensor(2434.0, dtype='float32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_9daca10fe89ca783762adb955d22f348(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb1fa392482923727e339b86e1855178
    def get_inputs(self):
        return [
            paddle.uniform([1045, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_6f63ba08863533b856c2b43717d04555(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c46c934312dd803dbe9571cdfed8240a
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_18da2c0985024563c9191e61258f01e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb1fa392482923727e339b86e1855178
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.2542722821235657], [-0.26889169216156006], [-0.3045753538608551], [-0.37000977993011475], [0.022664368152618408]], dtype='float32').reshape([5, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_c2c00f872f59bb32110f27151d83db50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb1fa392482923727e339b86e1855178
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.4553213119506836], [0.2287808656692505], [-0.3646608591079712], [-0.07116848230361938], [-0.07449036091566086]], dtype='float32').reshape([5, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_1e6ac30bbe0f47fa64f51914aa4f05c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bc7ea742aa14540ff5fdd6163035046
    def get_inputs(self):
        return [
            paddle.uniform([1, 1248], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_dcc53d2d6ee7de0723d50deb27fa05d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bc7ea742aa14540ff5fdd6163035046
    def get_inputs(self):
        return [
            paddle.uniform([171, 480], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_c08043e2311868d7a479d1bcdb6cbce7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_471e37956592534c29c856a2de770c75
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_3ed6967a73f460c0c24e630240889962(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bc7ea742aa14540ff5fdd6163035046
    def get_inputs(self):
        return [
            paddle.uniform([145, 36], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_e68a119239ac04e2f327fcbfef77b262(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb1fa392482923727e339b86e1855178
    def get_inputs(self):
        return [
            paddle.uniform([2335, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_97c44cf3e6d217fcfacd87bcfb330940(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c46c934312dd803dbe9571cdfed8240a
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_95df9ca8eebc1b8a5bb5946e0f50ee4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb1fa392482923727e339b86e1855178
    def get_inputs(self):
        return [
            paddle.uniform([2983, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_3113d5ac37fc41fbacac6567042dc390(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c46c934312dd803dbe9571cdfed8240a
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_356b61de89c1856d2064b2b17c18ec04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb1fa392482923727e339b86e1855178
    def get_inputs(self):
        return [
            paddle.uniform([3778, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_070f972b80a2d73df7da35d3db17bf6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c46c934312dd803dbe9571cdfed8240a
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_71286e2452d0e8f452f089b67e3d0b19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bc7ea742aa14540ff5fdd6163035046
    def get_inputs(self):
        return [
            paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_7c66e0b88812e05a8b91b5b55036eb87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bc7ea742aa14540ff5fdd6163035046
    def get_inputs(self):
        return [
            paddle.uniform([22, 480], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_e538f0c3f48196198de169707778036a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bc7ea742aa14540ff5fdd6163035046
    def get_inputs(self):
        return [
            paddle.uniform([145, 480], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_6fa4018a1fc3614904fa659fc7101454(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bc7ea742aa14540ff5fdd6163035046
    def get_inputs(self):
        return [
            paddle.uniform([171, 36], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_fd3e6fd06b5a40d399e9726e3d0c6179(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bc7ea742aa14540ff5fdd6163035046
    def get_inputs(self):
        return [
            paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_e98780bc2a7d52d5ba71faefcd7bdd43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb1fa392482923727e339b86e1855178
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.4088980555534363], [-0.0902586579322815], [-0.30522555112838745], [0.19477760791778564]], dtype='float32').reshape([4, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_961020ced8ac992ee1e5886b0bd43e5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb1fa392482923727e339b86e1855178
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.12573572993278503], [-0.41359490156173706], [0.02159428596496582], [-0.027848154306411743]], dtype='float32').reshape([4, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_72f2cb755d7678d9f979b2312e2d71a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb1fa392482923727e339b86e1855178
    def get_inputs(self):
        return [
            paddle.uniform([2064, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_8702952d4aae1b192ea67c4369722098(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_471e37956592534c29c856a2de770c75
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_aaf6267a66b962b70ebfe331f726d03a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb1fa392482923727e339b86e1855178
    def get_inputs(self):
        return [
            paddle.uniform([4223, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_8f26388358b96697c1347dbefb0975c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c46c934312dd803dbe9571cdfed8240a
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_099bd0cd0fb99db64281243bc85b01db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bc7ea742aa14540ff5fdd6163035046
    def get_inputs(self):
        return [
            paddle.uniform([1, 624], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_db59b7cd3c2a155b2cb806424d103ccd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.0
        input_2 = 1.0
        return paddle._C_ops.clip(input_0, min(input_1, input_2), max(input_1, input_2))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 72], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_38cd9eb6ad5af7b67b3c90007de38403(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db59b7cd3c2a155b2cb806424d103ccd
    def get_inputs(self):
        return [
            paddle.uniform([1, 72], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_ab75d3cd2d7f2f226e6100491cd40913(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.0
        input_2 = 1.0
        return paddle._C_ops.clip(input_0, min(input_1, input_2), max(input_1, input_2))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 92], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e2b1fad0a1be0bd97d8a545a20ea3daf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab75d3cd2d7f2f226e6100491cd40913
    def get_inputs(self):
        return [
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_6434ed3532730e39002c0ada0c6b5b77(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.0
        input_2 = 1.0
        return paddle._C_ops.clip(input_0, min(input_1, input_2), max(input_1, input_2))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 960], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e9b3ebb739d3dac2b5ea8cb51ac9953f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6434ed3532730e39002c0ada0c6b5b77
    def get_inputs(self):
        return [
            paddle.uniform([1, 960], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_cbc85e310b6b83b89b3feccb4b72c53b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.0
        input_2 = 1.0
        return paddle._C_ops.clip(input_0, min(input_1, input_2), max(input_1, input_2))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 480], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9e13ccfb8c2cb0d403b1ecf1ed54e258(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbc85e310b6b83b89b3feccb4b72c53b
    def get_inputs(self):
        return [
            paddle.uniform([1, 480], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_5dcbff40e22d4bcbd61e2c041ff85e39(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.0
        input_2 = 1.0
        return paddle._C_ops.clip(input_0, min(input_1, input_2), max(input_1, input_2))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 336], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b54066dd16993e1455a280ebd4d30a4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dcbff40e22d4bcbd61e2c041ff85e39
    def get_inputs(self):
        return [
            paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_4e66918b324062ee030df2198812fb6a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.0
        input_2 = 1.0
        return paddle._C_ops.clip(input_0, min(input_1, input_2), max(input_1, input_2))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_39f6ec89cfb8f9ad3ad84943b63c157b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e66918b324062ee030df2198812fb6a
    def get_inputs(self):
        return [
            paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_b7d96087b13a9784b8c287e2d270c47d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dcbff40e22d4bcbd61e2c041ff85e39
    def get_inputs(self):
        return [
            paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_80e1e9373b1bedd26291dc7193ef1cb0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.0
        input_2 = 1.0
        return paddle._C_ops.clip(input_0, min(input_1, input_2), max(input_1, input_2))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fe05765396121a9f2e036e96bab64c1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80e1e9373b1bedd26291dc7193ef1cb0
    def get_inputs(self):
        return [
            paddle.uniform([145, 240], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_c97a346a0a4afcf9f46b694ce59da5d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e66918b324062ee030df2198812fb6a
    def get_inputs(self):
        return [
            paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_0a40a18ccd42c856160909ab79e0282b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.0
        input_2 = 1.0
        return paddle._C_ops.clip(input_0, min(input_1, input_2), max(input_1, input_2))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 872], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3c1ddd0bfbba1a9be2d3ef13f8e99afd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a40a18ccd42c856160909ab79e0282b
    def get_inputs(self):
        return [
            paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_b97991662d746a1a96a4b6644c3b32c9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.0
        input_2 = 3.402820018375656e+38
        return paddle._C_ops.clip(input_0, min(input_1, input_2), max(input_1, input_2))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e82d824f8240d8478ac197bc6f9fe1ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b97991662d746a1a96a4b6644c3b32c9
    def get_inputs(self):
        return [
            paddle.uniform([1812, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_20011133820f9d2cd7532b74a85ea530(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dcbff40e22d4bcbd61e2c041ff85e39
    def get_inputs(self):
        return [
            paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_48dcd163bcbf6596e67b6609662924ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80e1e9373b1bedd26291dc7193ef1cb0
    def get_inputs(self):
        return [
            paddle.uniform([10, 240], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_4d9da502945a986be48dc1e44d78fc25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b97991662d746a1a96a4b6644c3b32c9
    def get_inputs(self):
        return [
            paddle.uniform([5519, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_15d97d87f7a9d9dd09ef464ee1f0e2de(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.0
        input_2 = 1.0
        return paddle._C_ops.clip(input_0, min(input_1, input_2), max(input_1, input_2))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cce736a372ceed1e92bf877435a4384a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15d97d87f7a9d9dd09ef464ee1f0e2de
    def get_inputs(self):
        return [
            paddle.uniform([10, 36], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_eae7978a657223c15d188c0f21d6d694(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbc85e310b6b83b89b3feccb4b72c53b
    def get_inputs(self):
        return [
            paddle.uniform([10, 480], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_c3ac76b56db83fcbd9a189ae7057dadb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b97991662d746a1a96a4b6644c3b32c9
    def get_inputs(self):
        return [
            paddle.uniform([1798, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_f3b377f7bf4a7d9d9531e6e1a6e1fac4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dcbff40e22d4bcbd61e2c041ff85e39
    def get_inputs(self):
        return [
            paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_8e953e8bf40c433bd3841ee9d5279ac4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.0
        input_2 = 1.0
        return paddle._C_ops.clip(input_0, min(input_1, input_2), max(input_1, input_2))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 100, 2], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_81cf4c0397f836cbc83fb1c49a559235(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e953e8bf40c433bd3841ee9d5279ac4
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 100, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_e11e99f53acd7f525126e75aa4c0a2ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80e1e9373b1bedd26291dc7193ef1cb0
    def get_inputs(self):
        return [
            paddle.uniform([171, 240], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_c87f86f048a09bdf5202f954e2bd2dcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e66918b324062ee030df2198812fb6a
    def get_inputs(self):
        return [
            paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_4853ad848829f4895968c7352c332867(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b97991662d746a1a96a4b6644c3b32c9
    def get_inputs(self):
        return [
            paddle.uniform([1574, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_5718acf341f713f3b4c5769d4c6f3869(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80e1e9373b1bedd26291dc7193ef1cb0
    def get_inputs(self):
        return [
            paddle.uniform([22, 240], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_2439e066ba63f6523c3f046590e641be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15d97d87f7a9d9dd09ef464ee1f0e2de
    def get_inputs(self):
        return [
            paddle.uniform([22, 36], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_847c4f5b665a535de6b3ca0701bc12c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e66918b324062ee030df2198812fb6a
    def get_inputs(self):
        return [
            paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_731658ac44f9187c30a4bd3392343393(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.0
        input_2 = 1.0
        return paddle._C_ops.clip(input_0, min(input_1, input_2), max(input_1, input_2))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 672], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e768885b630fe0bd45943ca7dfc031c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_731658ac44f9187c30a4bd3392343393
    def get_inputs(self):
        return [
            paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_7ffd6d8c02a8388385127edafe974729(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b97991662d746a1a96a4b6644c3b32c9
    def get_inputs(self):
        return [
            paddle.uniform([2055, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_818e2e251e2b54d7bb9be5ce21cdcccc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b97991662d746a1a96a4b6644c3b32c9
    def get_inputs(self):
        return [
            paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_a1aff4d74c16449ef91c19bbfacf4d67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b97991662d746a1a96a4b6644c3b32c9
    def get_inputs(self):
        return [
            paddle.uniform([1045, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_ec18cc6a49b270baf5e9bb651e864506(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.0
        input_2 = 1.0
        return paddle._C_ops.clip(input_0, min(input_1, input_2), max(input_1, input_2))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1248], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a3427548cc5b33d6365f91a55a192574(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec18cc6a49b270baf5e9bb651e864506
    def get_inputs(self):
        return [
            paddle.uniform([1, 1248], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_c30f0ea4f6be81b8859c7a00f13d84e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbc85e310b6b83b89b3feccb4b72c53b
    def get_inputs(self):
        return [
            paddle.uniform([171, 480], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_a9ad55724cdcf5a34152fd2756a255e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15d97d87f7a9d9dd09ef464ee1f0e2de
    def get_inputs(self):
        return [
            paddle.uniform([145, 36], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_a4a3a714947268312866e43f3ac55264(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b97991662d746a1a96a4b6644c3b32c9
    def get_inputs(self):
        return [
            paddle.uniform([2335, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_9f224483da9ee5006e76e1d7f42a3fe9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b97991662d746a1a96a4b6644c3b32c9
    def get_inputs(self):
        return [
            paddle.uniform([2983, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_75d96c1308a76b6db94d09d4edcd408b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b97991662d746a1a96a4b6644c3b32c9
    def get_inputs(self):
        return [
            paddle.uniform([3778, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_f45f5366be2473a9f768d95096fbf648(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.0
        input_2 = 1.0
        return paddle._C_ops.clip(input_0, min(input_1, input_2), max(input_1, input_2))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 156], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8068e88e1db045583ce52b8d60512223(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f45f5366be2473a9f768d95096fbf648
    def get_inputs(self):
        return [
            paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_5499127f4c1ece5a79c603e2fecad21f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbc85e310b6b83b89b3feccb4b72c53b
    def get_inputs(self):
        return [
            paddle.uniform([22, 480], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_0668543b79e5b68c1766e275d39d4cc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbc85e310b6b83b89b3feccb4b72c53b
    def get_inputs(self):
        return [
            paddle.uniform([145, 480], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_110e3426636b70e9e2edc898ed5ae9d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_15d97d87f7a9d9dd09ef464ee1f0e2de
    def get_inputs(self):
        return [
            paddle.uniform([171, 36], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_55af9a773d6695abc74080c466fed7a2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.0
        input_2 = 1.0
        return paddle._C_ops.clip(input_0, min(input_1, input_2), max(input_1, input_2))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 120], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1c9e429b37332e5b54e7b141026446bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55af9a773d6695abc74080c466fed7a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_c537fd4ce29a13046f037be93d4baabc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b97991662d746a1a96a4b6644c3b32c9
    def get_inputs(self):
        return [
            paddle.uniform([2064, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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
class TestPrimitiveOp_7d900db1214ae484a8b32f32c500b31e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b97991662d746a1a96a4b6644c3b32c9
    def get_inputs(self):
        return [
            paddle.uniform([4223, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

class PrimitiveOp_1f7cc33e7648386579db633f2b112f5c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 0.0
        input_2 = 1.0
        return paddle._C_ops.clip(input_0, min(input_1, input_2), max(input_1, input_2))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 624], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7ae5cb2d248e42d5eca5b690ae5aa698(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f7cc33e7648386579db633f2b112f5c
    def get_inputs(self):
        return [
            paddle.uniform([1, 624], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
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