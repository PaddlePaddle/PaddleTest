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
class PrimitiveOp_57a08b40f1cc5fad88c48321abf511c4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1f8679b7ef3a1d529b8cdf365a3ebee3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57a08b40f1cc5fad88c48321abf511c4
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor(160, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_779dd1a1352b01918a83b3635ab6a202(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ae0a426f5487bb92257c84b9eb804bb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 128], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 128, 128, 128], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3e815ed69aecb8254a26d6a2335b407e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 32], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 128, 32, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e1661351f1084edd98a799df6b299a4a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f73c38b529391edceff98fc3da0587b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1661351f1084edd98a799df6b299a4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 26, 512], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 26, 512], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4b7aa4f538da06eae5402578d78a8aec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6c4b0af8495d3f634b6c75ce53f2961f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 16, 16], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_346442fe6e1318d2f35eeedfa66b1173(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 128, 128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d3b00a86147f37c75b8d3ca7ed53c035(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 32, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4989ad62757d0df63e3b78a428bf143f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57a08b40f1cc5fad88c48321abf511c4
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor(240, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_afb9e67dffd391c3910f560c49561836(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 120, 120], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d0a81e250d273a22a6c907cdfbcbb996(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 120, 120], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_debfad2651c2b6dc1e66b9b3dbc51737(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 60, 60], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5143a909b8f812b811e10708f65e2176(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 60, 60], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_45ccc4b39bde0c7999218e6ac6920f55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 30, 30], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ed13ecdfd61c39554b594e877a9bfbd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 30, 30], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_22cc9f1a0c568788b5ed488193f05f7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6c1d1bff5ec90f6139e9c706b4c5837d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0e01966addf5f909dcea226606e2bb93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 80, 80], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_45f679ae06b79926fdc4327b94d40b34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 800, 1344], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([[[[0.071533203125]], [[0.231689453125]], [[0.33837890625]]]], dtype='float16').reshape([1, 3, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2667b5e6e9756e7a3290c67ef5d54116(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0f5e1d5966af75d11f446038346c6a50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2667b5e6e9756e7a3290c67ef5d54116
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 100, 168], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2371826171875], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cc4f7afa3198caf9c0d2d1e6724a8585(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2667b5e6e9756e7a3290c67ef5d54116
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 50, 84], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.09674072265625], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ad7dfce75f1c6252933a11f4120bcecf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2667b5e6e9756e7a3290c67ef5d54116
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 25, 42], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1575927734375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d238c3322216659778541d050726688e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2667b5e6e9756e7a3290c67ef5d54116
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 13, 21], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.08050537109375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_90c801091364aee773b52fd8c7c13cb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2667b5e6e9756e7a3290c67ef5d54116
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 7, 11], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.310302734375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3f51fed851bdfeb9b1fdb9c076b3c0e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1661351f1084edd98a799df6b299a4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 16800, 80], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 16800, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dd0cde618ee6417d7fd7fa57d38eef78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1661351f1084edd98a799df6b299a4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4200, 80], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 4200, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_adda03496921eafbda0402347f4f1a6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1661351f1084edd98a799df6b299a4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 1050, 80], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1050, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_aaac80cafce3bfd849c9ebc59bd29cfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1661351f1084edd98a799df6b299a4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 273, 80], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 273, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dd3cfd80e1ad2c96f4342e2eb97634e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1661351f1084edd98a799df6b299a4a
    def get_inputs(self):
        return [
            paddle.uniform([1, 77, 80], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 77, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e4fef97895a7984e150945b02090c402(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 128, 64], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_8521d29e3b784d37436de85b5d72a933(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 128, 64], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_b0bade2580f9db9bb121285d164280b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 64, 32], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_d78b7ad2366e3fb856dc37fee3cf4593(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 64, 32], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_f045f09cdf0898ebfd1e735cb58b5104(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 32, 16], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_ea40c97cd3708fcfb7ca4467dbbddb75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 32, 16], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_d15c3b9d5411a7b61f88093d6edbf0bf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_028ff1678d35a39d91b451bab6f2c099(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d15c3b9d5411a7b61f88093d6edbf0bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 1, 9, 112, 112], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 2, 16, 9, 112, 112], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a44a9a52ffae8667c6729359ca111fd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d15c3b9d5411a7b61f88093d6edbf0bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1, 49, 56, 56], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 4, 16, 49, 56, 56], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d746f0d6b73e4421e7ada7689a970641(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d15c3b9d5411a7b61f88093d6edbf0bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1, 49, 28, 28], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 8, 16, 49, 28, 28], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_18c23ca6bb6ab6b014232e711ef504b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d15c3b9d5411a7b61f88093d6edbf0bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 1, 49, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 16, 16, 49, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4415a4d757eb4998281c2b4335b4b819(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d15c3b9d5411a7b61f88093d6edbf0bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 49, 7, 7], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 32, 16, 49, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_de886f70d0ec008050e6b55c840fe448(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4784398078918457], dtype='float32').reshape([1]),
            paddle.uniform([1, 16, 480, 480], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ee0bc621b9b90b8687ba04fa3c0993f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.14870469272136688], dtype='float32').reshape([1]),
            paddle.uniform([1, 16, 480, 480], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a760ad83b1797b51495cb16a085b78c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1273345649242401], dtype='float32').reshape([1]),
            paddle.uniform([1, 32, 480, 480], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_584e9910c02ffb52234403d5fae7ab8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.011404834687709808], dtype='float32').reshape([1]),
            paddle.uniform([1, 32, 480, 480], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3af97ed9c2317002af8019219f9d4c55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4188338816165924], dtype='float32').reshape([1]),
            paddle.uniform([1, 32, 240, 240], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d36fa7ad7250de65df7ddaeac278ed06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4657136797904968], dtype='float32').reshape([1]),
            paddle.uniform([1, 48, 240, 240], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dde6cb18f45cdc58556b1d5f8d139aeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.13037486374378204], dtype='float32').reshape([1]),
            paddle.uniform([1, 48, 240, 240], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3a7584c06e84e7e63bcf0efa16f3a4cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1505524218082428], dtype='float32').reshape([1]),
            paddle.uniform([1, 48, 240, 240], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7b9d3513892a89fdcff0e2b2faa30ab8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0886026993393898], dtype='float32').reshape([1]),
            paddle.uniform([1, 48, 240, 240], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_37e116d91b54ab26cd83882e4976556b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.17881451547145844], dtype='float32').reshape([1]),
            paddle.uniform([1, 48, 240, 240], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4fd27c13ac943983a960d3b2426ed22d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4926683306694031], dtype='float32').reshape([1]),
            paddle.uniform([1, 48, 240, 240], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c0aba6e71f57e10628aa914a6c7b5d6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.329205721616745], dtype='float32').reshape([1]),
            paddle.uniform([1, 48, 120, 120], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5b95cce4b9f7733bb0be8129d8253cf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.43924281001091003], dtype='float32').reshape([1]),
            paddle.uniform([1, 96, 120, 120], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5622aca1ab62caf843d0cd9a93e81f56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.05343525856733322], dtype='float32').reshape([1]),
            paddle.uniform([1, 96, 120, 120], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_274f0e8098d13c185451dbe23141c81a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.41747212409973145], dtype='float32').reshape([1]),
            paddle.uniform([1, 96, 120, 120], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0ee3cab7935ebdc06da4bf17f4476265(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.13127554953098297], dtype='float32').reshape([1]),
            paddle.uniform([1, 96, 120, 120], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7980209911d66731d173400fe08683b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.49068695306777954], dtype='float32').reshape([1]),
            paddle.uniform([1, 96, 120, 120], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4e7f9650f707d718d121ca520f55a3c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.11958789825439453], dtype='float32').reshape([1]),
            paddle.uniform([1, 96, 120, 120], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4a063da814de163cb11a4ca0e14bb048(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.34703755378723145], dtype='float32').reshape([1]),
            paddle.uniform([1, 96, 60, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7afb730baef295c7c38c837d26c1e486(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4826791286468506], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b40495a1c0ec5171699f023317a4547f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4372650384902954], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_58b6a5c5fdae8d1a4d0b07ff0f2c617e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.13426004350185394], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_13419c44ad7aebaee3d22dc289fd8411(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.20082494616508484], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0fbde7be8cad9a47e10e689d49f6f013(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1471903771162033], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ae1f5f1555c32f751df6248b87cdf0c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.19031450152397156], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_88b5b9e8b0f2d80e22c7311e30efd7d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.23371702432632446], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e59220d918f90d9486305542316eb590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.460300475358963], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cbe53994399bf243c4ca2dd54cd48cf1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.14110355079174042], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3567ee417eb440cb5e0026cefd78c3c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10759323835372925], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a74d8beb22034919ee0fd7535d52a646(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.13945358991622925], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_27c9fc4ed7a40a86ed949563622ea172(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1550578624010086], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_036bb8fd6ebc7466da9bd6e18f9e7603(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.18472763895988464], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_26f62ab760e57c40f2e62266fa5a24ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.48091572523117065], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b73b9932aba8f980bc2230d57116c1da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.40070024132728577], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_60119f9d2432d25461ae29a7b3f95cc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.03564074635505676], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c49ef462e444d79ab63eeaf3e9955df1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.14680147171020508], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_170deafac1a8b37567e117a0e45d7a5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.46726563572883606], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cc2f4cc51ebf893793605c08ec80212c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.09462948143482208], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 30, 30], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1eb1db9325e4aa293c1853376d0f1d15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 30, 30], dtype='float32', min=0, max=0.5),
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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2a8541190f54ae4288b6f850f9c33b48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.08917675912380219], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e5b0930782592a8c609675b9c4c3be4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10074345767498016], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_baeaaf5b79d2d7d639b2ca939894a5be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.34102490544319153], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_06871689df0ce1db1834eed66bd566fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3246842622756958], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1c5a3cee189ccff751fc4eb0bd67548f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 30, 30], dtype='float32', min=0, max=0.5),
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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0275a0775a372a379ef80109bb063925(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3248758912086487], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_90a572f4dacf09b84f534df6a4b46082(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.20371150970458984], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_391b216546304e096e920ae4b499e81d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.028545115143060684], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1c4a952fe05f20502cb397cbdcd7ce04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.26297762989997864], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8ce81bf413426596e5e06cd45515b6bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.06808995455503464], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e925a0f497e73f1c2a1530443aab013a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1499202996492386], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3bc4bbaba16abfdb57a5a17763b58ce3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.044834017753601074], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_454925d8843847985109fe6f44687e8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.21141599118709564], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a5ce12d1e52c71242bf9fe7d31130f7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.31889456510543823], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_195d36ddead39ed49240c8a7db45a08a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.24018998444080353], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_997175afa5074f1ce83aaaf5f8ac9f97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 30, 30], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_25a3eb512cd312a7296e3e1d80526983(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 60, 60], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_094030fcf6989c7b0e628d3d2b3d0daa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 120, 120], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_33c11b3ede60b50bc83239014a276487(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 240, 240], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ce45a57708dfcc328970d6d64898c10f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_07af2e9ebc35d7ba65ba8f9bd5042f63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_72e4cb4835adbc1c3a6d816cbf23534e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 120, 120], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2d3902089199c1453cff6f1757499550(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 240, 240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2fad5ef2dd276157c7e575e0ebd2282a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_acb42c1669512ad565fe3e4d746012a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2fad5ef2dd276157c7e575e0ebd2282a
    def get_inputs(self):
        return [
            paddle.uniform([32, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 32, 56, 56], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8dbd249d287671692e8c2d3b460dd760(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2fad5ef2dd276157c7e575e0ebd2282a
    def get_inputs(self):
        return [
            paddle.uniform([64, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 64, 28, 28], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_07ca3cc068d60040fb2568770d408c5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2fad5ef2dd276157c7e575e0ebd2282a
    def get_inputs(self):
        return [
            paddle.uniform([160, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 160, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f82704d6b1b49bb3de719b55d46518e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2fad5ef2dd276157c7e575e0ebd2282a
    def get_inputs(self):
        return [
            paddle.uniform([256, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 256, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_63cb82959f16fcbe33b2fe44a26691f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 28, 28], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_eca3ef6c17f90fdee50aa0c6aad3f8ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 14, 14], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_a34094f090ba0e65f6b83302827c1aef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 14, 14], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_7d7e93da71ee65ac503620a33ca2a6d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 14, 14], dtype='float16', min=0, max=0.5),
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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c24ba67d0c01b6b1009b412b2b47240b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 7, 7], dtype='float16', min=0, max=0.5),
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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dfb6bd0bab173c18ba570711b98e8401(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 7, 7], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e01383c0dee9f5741384a4a0d086fb8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.319091796875], dtype='float16').reshape([1]),
            paddle.uniform([1, 16, 480, 480], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bd096d926f6ddb2bed55759f661318bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.337890625], dtype='float16').reshape([1]),
            paddle.uniform([1, 16, 480, 480], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b73bfd8ba2ab715f6d33d0171219da30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0692138671875], dtype='float16').reshape([1]),
            paddle.uniform([1, 32, 480, 480], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_550ba12270dbb2265a80c39b62b04a6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2271728515625], dtype='float16').reshape([1]),
            paddle.uniform([1, 32, 480, 480], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0a096f6a6d2862efe3258abde200e323(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.459716796875], dtype='float16').reshape([1]),
            paddle.uniform([1, 32, 240, 240], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_452253d66f2d074c7a143dd9f19a463f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.267578125], dtype='float16').reshape([1]),
            paddle.uniform([1, 48, 240, 240], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_42bff6e60368b2fde2f3276afee9561e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0087127685546875], dtype='float16').reshape([1]),
            paddle.uniform([1, 48, 240, 240], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_030142ae21426be869fb4ce587db8a01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.369140625], dtype='float16').reshape([1]),
            paddle.uniform([1, 48, 240, 240], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9e3985a1772602c98cb5949759e5cd90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.007537841796875], dtype='float16').reshape([1]),
            paddle.uniform([1, 48, 240, 240], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d13ffb68a5d24c7ead5331d6e8d033f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.078125], dtype='float16').reshape([1]),
            paddle.uniform([1, 48, 240, 240], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4bd4e6819d9ea021a8a9a634d0e6fe46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.128662109375], dtype='float16').reshape([1]),
            paddle.uniform([1, 48, 240, 240], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7889092a20fb7b6ba7669496b6f9ef8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1458740234375], dtype='float16').reshape([1]),
            paddle.uniform([1, 48, 120, 120], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_db450c183f62bbfa8458e4f6508c71f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.267578125], dtype='float16').reshape([1]),
            paddle.uniform([1, 96, 120, 120], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dd3705a24261f165ede83d38fcd2c593(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.395263671875], dtype='float16').reshape([1]),
            paddle.uniform([1, 96, 120, 120], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4a45b0197736dc141a06698fbd54fabc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.191650390625], dtype='float16').reshape([1]),
            paddle.uniform([1, 96, 120, 120], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b22360feea71613f40bda76c42e67329(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.19287109375], dtype='float16').reshape([1]),
            paddle.uniform([1, 96, 120, 120], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d70acd5162d0ecde8425b5bcdd0d6bc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4609375], dtype='float16').reshape([1]),
            paddle.uniform([1, 96, 120, 120], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1a2e2b1d3818e1628752ad9f337bd6d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.187255859375], dtype='float16').reshape([1]),
            paddle.uniform([1, 96, 120, 120], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0e333d3b996c5ae1dc6711290e43b251(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1800537109375], dtype='float16').reshape([1]),
            paddle.uniform([1, 96, 60, 60], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6ec4120da0d7ba0d4a7ebde670733363(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.43896484375], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b4b63461613deeb3eeeb0eba744258e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.156005859375], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_655319964955ba6592f13f7848cda410(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.48974609375], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cb30f6306a53a0097f191ffe5264f0c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.451171875], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_52a7216a38a8a7c38b7c049e8fe18e0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.384521484375], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dc0b58019cf3763683b93034aa168d05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.352294921875], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_12ce307862a10e8c6e23baf29725d67e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.249267578125], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0aa55745543d8d0806928533f0cdaf36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1373291015625], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e1b49181d3764f4f80f54113c3b2bd43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.423828125], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3a6cacb4996ad0798d4c4dee662e89a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.234619140625], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1d295c3209cc20da3e597604e456874c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3466796875], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2cf101a4bf6be92436effe5c980d7078(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.346435546875], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_724f4ca24c680eac73170e4dd4dcdea0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3779296875], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0247687308f8ae4c8c4f785aaf06ef79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.11480712890625], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7de1d52acdd9232ac621a71bf2263b10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0653076171875], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4cce55e847026f1192cdd6c47812536c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.357177734375], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c6e787ad791f95dcbacdb04771774601(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3251953125], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e0d61aaa9442d34c264313b07b58506b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.408203125], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fe70e85a18d0f9e8be7438ea0984cebe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1578369140625], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 30, 30], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7412ba9ca6ced045ade123f901c8c329(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 30, 30], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_b3e7ab7e192553a6dd291c31c74974ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0192108154296875], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e1d564047965903e86a10bd90905d6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.03240966796875], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b7f49611a77fe31625a06c52a5aa7af9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.280029296875], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d9c5fe69d82c53131770314212804ad9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.421875], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c8430c22c35a324d2fb7dcca19c09ea6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 30, 30], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_6743d2884bbd2ae9711f412f7dcb14d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.027099609375], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_adc94c001c1f972a57ac30feace21322(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.461181640625], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d97ccbfd814a7bb4421c76db1e373627(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.348388671875], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_37115abae1889c20ac57f06fc239ccce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.34423828125], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fb98f1254a0f2bdd6c8514aa945bc1ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.281494140625], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7532bb20647a867b75652525256e502c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.09295654296875], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_887d263b6f43ef0d41490471c726c748(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2174072265625], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_39109c24d142c7ec2eaa85ee08d9b84a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.22265625], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0e9e7bcdfffea558b65ef43b25db61e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.476806640625], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_98ccab4c6ca09d9e483db2d5925bd7f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3193359375], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0db5d1618893c796ad43e0fb4fede6ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 30, 30], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_5753c0b2988381f81bff5cdcb886ac0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 60, 60], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_101f12d799cb4f6e4f684c82b7fab96a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 120, 120], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_f00ca7dc0374ec449e8147d47348aaa2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 240, 240], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_b030b5948b91942ffdcd0788b3f02f6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 30, 30], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float16').reshape([1, 24, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_40f2ab723f92ef5b4e422ef7c3b7a1fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 60, 60], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float16').reshape([1, 24, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0dc578cba7a8f132f770fe2fd626fd8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 120, 120], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float16').reshape([1, 24, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_315d65f36e27ddb8dc5648785a3084a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 240, 240], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float16').reshape([1, 24, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3c5705f7e47b5c95a226810ed45ad6fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 128, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_744b677c09f2d7ec37e7a4fe173c3a7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 128, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a41a6b41b53796b65ca1ea09fbf0dbdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 64, 32], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a3cfd414c774e15ce4af23443f7984b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 64, 32], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_813c87e4f6340800e8d32460970cf5d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 32, 16], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_56cd587e23929a2d27210cea691e7f25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 32, 16], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4659a607e846f7a9d43bc6d2b829ad98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 26], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 26], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b04ae32913218b3e74af7baada94fdd9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0421f19f9d9fa9a3106909bb8dafb19a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b04ae32913218b3e74af7baada94fdd9
    def get_inputs(self):
        return [
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 196, 320], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9d1979033f4e6b8b5785adf23c7e8a47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b04ae32913218b3e74af7baada94fdd9
    def get_inputs(self):
        return [
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 49, 512], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_32ca0dcad8d739fe5c1d20410f860131(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 26], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d3e60a9910a72ef31a8c3a4b705269c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 120, 120], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_7b6ed0d0432f9a44f37429dbbfce6d7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 120, 120], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_f7d091e6c32b87a8fabc92543d6eaee8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 60, 60], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_85edd5e8f9687b02b0a06e0b74319283(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 60, 60], dtype='float16', min=0, max=0.5),
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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3c80abf9830843485286b71737542974(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 30, 30], dtype='float16', min=0, max=0.5),
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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_491693e2f204932150ad463fc1d80432(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 30, 30], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_8d4ba7491b733b477faf2b62e7ed67ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 800, 1344], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.07380150258541107]], [[0.3107784688472748]], [[0.39664655923843384]]]], dtype='float32').reshape([1, 3, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8e09101bf720e92d8e0f1d4b16a4959d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f02e63e18ac0a063e16cc329b4ada23c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e09101bf720e92d8e0f1d4b16a4959d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 100, 168], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.44416195154190063], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7ddd9c751585f3710b702ddce02bd329(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e09101bf720e92d8e0f1d4b16a4959d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 50, 84], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.41823989152908325], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ead475da8d193c5fb6755eaabb50be1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e09101bf720e92d8e0f1d4b16a4959d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 25, 42], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.31520557403564453], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_98a390b7f24fecb6e04cecc3d4d86d3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e09101bf720e92d8e0f1d4b16a4959d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 13, 21], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4658982455730438], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_52bda404e1425ca9429c95f6e85bd915(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8e09101bf720e92d8e0f1d4b16a4959d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 7, 11], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4570605158805847], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_68d8b82cdbc57005a461b47dfc15c16d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f49158b226088c5dd7c8d96d7698e861(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_68d8b82cdbc57005a461b47dfc15c16d
    def get_inputs(self):
        return [
            paddle.uniform([1, 16800, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 16800, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_81ae58da920f3596fc9fb35afa397a0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_68d8b82cdbc57005a461b47dfc15c16d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4200, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4200, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4a8eee5eeb1e55e1a5c61fbb6e211f0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_68d8b82cdbc57005a461b47dfc15c16d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1050, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1050, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_edeb650fd5cc6111ffaf52ee26439bcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_68d8b82cdbc57005a461b47dfc15c16d
    def get_inputs(self):
        return [
            paddle.uniform([1, 273, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 273, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_23ce39129ceb71c76592fc715979e6e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_68d8b82cdbc57005a461b47dfc15c16d
    def get_inputs(self):
        return [
            paddle.uniform([1, 77, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 77, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_618b12e259d6f7fa16e5850bc8691fa8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 28, 28], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_019189daa560c8c5f963ba979857bf58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1e405e4a460b50cbb81154dc367c05aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 14, 14], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6d1380c104818c8f5d7668299d2cd496(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 14, 14], dtype='float32', min=0, max=0.5),
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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3862fb48df379274c1679a73b1ac6186(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 7, 7], dtype='float32', min=0, max=0.5),
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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cbe7cc8985244c47df3897d91fab6678(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 7, 7], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_37d4e7659c024af2f567130359fa8779(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_932886dee0198d3c0c7a838e3718160a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37d4e7659c024af2f567130359fa8779
    def get_inputs(self):
        return [
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 196, 320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_42b5f62095e34048db648ed80442598e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37d4e7659c024af2f567130359fa8779
    def get_inputs(self):
        return [
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 49, 512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1d5fd3171471f8e2cac8e37dfed6f609(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_68d8b82cdbc57005a461b47dfc15c16d
    def get_inputs(self):
        return [
            paddle.uniform([1, 26, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 26, 512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b760ff468d2c7f737fb1319c95be0b64(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e08cda28bee4cedb22a4dd3ac8c846bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b760ff468d2c7f737fb1319c95be0b64
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 1, 9, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 16, 9, 112, 112], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a7cc76129723b1f7fc421045e63f3cba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b760ff468d2c7f737fb1319c95be0b64
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1, 49, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4, 16, 49, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4dae431f63189dd7a4ac441918092553(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b760ff468d2c7f737fb1319c95be0b64
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1, 49, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_925144f8b217c0d32051ca16d7d99079(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b760ff468d2c7f737fb1319c95be0b64
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 1, 49, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5f738fc38f6f75d4737b7e88db59c632(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b760ff468d2c7f737fb1319c95be0b64
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 49, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f008fb874e28b27a543ece6c847b21b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 16, 16], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_23fd76a132333f9be863ffc51f85c949(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 16, 16], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_1c877cebcf5f6d38d2719c13ead3dc62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.29906246066093445], dtype='float32').reshape([1]),
            paddle.uniform([1, 16, 320, 320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3cf758f462707869752567beecc2ffad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.43402084708213806], dtype='float32').reshape([1]),
            paddle.uniform([1, 16, 320, 320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dadeed65ee472554044ee18ff3d1dbe6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2055632621049881], dtype='float32').reshape([1]),
            paddle.uniform([1, 32, 320, 320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6fbdf76eb1e951bc10f9ca2c84c00cbe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.22573179006576538], dtype='float32').reshape([1]),
            paddle.uniform([1, 32, 320, 320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_774e31cd113a2d916230760f015fd77a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.42056146264076233], dtype='float32').reshape([1]),
            paddle.uniform([1, 32, 160, 160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1d5ff4ed69aa5c2a489df8fb3273d410(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.12126166373491287], dtype='float32').reshape([1]),
            paddle.uniform([1, 48, 160, 160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9d1b9e02047d2a9bd052a41bc2b1503a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.09242968261241913], dtype='float32').reshape([1]),
            paddle.uniform([1, 48, 160, 160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f8964ac032bc131dfed47ac85e3ca33e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.004355288110673428], dtype='float32').reshape([1]),
            paddle.uniform([1, 48, 160, 160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_61dd32bd12b6ab894bcdd5b3ecfabfd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.11184486001729965], dtype='float32').reshape([1]),
            paddle.uniform([1, 48, 160, 160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2f6afc91fb76920debd10d1ee53be078(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.14176878333091736], dtype='float32').reshape([1]),
            paddle.uniform([1, 48, 160, 160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_36d41e303411e00e12765d014e4be292(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.23379838466644287], dtype='float32').reshape([1]),
            paddle.uniform([1, 48, 160, 160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ce047e84c303048a5cb3d3fba55f05c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3178041875362396], dtype='float32').reshape([1]),
            paddle.uniform([1, 48, 80, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8cc4b84351f122d69d458e1f440b77c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.15914151072502136], dtype='float32').reshape([1]),
            paddle.uniform([1, 96, 80, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_aeeafc1200bab01ce7b6d4657c6c0dd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.34217628836631775], dtype='float32').reshape([1]),
            paddle.uniform([1, 96, 80, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9813ec5c8742ffa5fa573066f1848288(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.21943950653076172], dtype='float32').reshape([1]),
            paddle.uniform([1, 96, 80, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d1ac6002153867e144178d7235b70a3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.046651214361190796], dtype='float32').reshape([1]),
            paddle.uniform([1, 96, 80, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8d66fbb8902116f06ace4e3f8bddabbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.23213328421115875], dtype='float32').reshape([1]),
            paddle.uniform([1, 96, 80, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b03ebf4bcda7ad4cf82567bce485371c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3153538405895233], dtype='float32').reshape([1]),
            paddle.uniform([1, 96, 80, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e20740fc3f78766a6cd66cbab9787b42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.09391975402832031], dtype='float32').reshape([1]),
            paddle.uniform([1, 96, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ce7693ea22127a06250af707e66b23c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.07466598600149155], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_467af8a5989fd8fc7d623ab5df778653(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.358701229095459], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_38ff51d7075d18096c1876962007f5ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.09427876770496368], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_040b49f4b9b917da05af914b3aa68466(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.49441027641296387], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_13e82fa5ec9ee2911797bb2f56ffd168(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.46217530965805054], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_52b6c919964993d92fe546194ef0e256(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0662069320678711], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d0e5ff6fd7b58b71d03aa7fc8e2b160b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1861724704504013], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_276c81df98e9bc685e3d2208e76260cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.06581340730190277], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c5f1d3d56e02be9a2838bab1294ecdcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.06165476143360138], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_defd4e7dcf970743870bd3c233c4e7e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.47823676466941833], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6b14a42117c96eaf0ec624555d3a0412(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.46561551094055176], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2b9fd63aa75a70b07d3dbd2e6a694009(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1863097995519638], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_334c7f7302a3bb2a62a5a15abdf75f9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.056068096309900284], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b0b47de37821e62e2dec860544f46ebd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3534022569656372], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_45b04a09984fda244d33f22c1e2e469f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.07152021676301956], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ca794122085f75cea3874a4615e8066f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2648335099220276], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_49656686ba87022c08123375abc87988(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.38199907541275024], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_387d03071ba4dd86db21fb7e657149ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.07299169152975082], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e5654ad6e033a84717518e7b501c47c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3615790903568268], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4c5368fc67b9418fad6148eebe8f95e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bd5740f61e2b827532565aa21c92934d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.21278151869773865], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a04b83938f9c2c10def4576094e4b11f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.34000837802886963], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_841050a796d4882aeb71037a808caf7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.31481823325157166], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fc650fc4fd64a7f00b97542165dfd70e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.39017990231513977], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c0953f8f54823485c469e0681bbb49ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2968728840351105], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_475bb72ea8c897a524b304c9fa6b803d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.046320926398038864], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_014f59bcb2bd6a5c57ab3c8543bc0ad3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.20497341454029083], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_34307e4be8620e8095d37149cbf0959f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.30073490738868713], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c29f171865ec7e528a2936a7b3a46169(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.49956014752388], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4c877feccd4531c14f71287cf4c77b03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.14100289344787598], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b57d6b4213d95ebce61df16e0bae7a25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3750498294830322], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1a19310b59fe3e72e1e30666059f481c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.24916493892669678], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4271cfc656941565270252b90b38b3d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10982408374547958], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_05c3cbf44355b1f2351025c093d14f2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe6317ab5a11427a07d0c2fccba71197
    def get_inputs(self):
        return [
            paddle.to_tensor([0.34541022777557373], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e5308d15235e59909c6185b93dc8754c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 20, 20], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_98a5f431764a8b2fbe9973a3abd66e5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 40, 40], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_09fa07eda11812be51fcfc9ff238b8de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 160, 160], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d3155b7e0d6336e09935e25c81b85a5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_05cdc975a73b86b66291163a1200ab40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bdf5eb086bb8c975838aba46edd523f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 80, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d3d13130fd6119f7aeae160c52bd3144(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 160, 160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4fc1311ccf1f6b0c124992b6a74c9ce1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 80, 80], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_5036a3755e18f60c49ce477350e7a198(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 80, 80], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_2e15e88791e5da6fec940e22f1e0662e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 40, 40], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_62563cd94c8d8fa37a75ec213d3862f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 40, 40], dtype='float16', min=0, max=0.5),
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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_124d15dfb2f18927710eb88d01bee039(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 20, 20], dtype='float16', min=0, max=0.5),
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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9bbb277ef56cac819f5adcea8d0e5333(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 20, 20], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_640b4044819ffc78459f2eceed8095ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.38720703125], dtype='float16').reshape([1]),
            paddle.uniform([1, 16, 320, 320], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2879999a38165f86b9b67df726c05408(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.09393310546875], dtype='float16').reshape([1]),
            paddle.uniform([1, 16, 320, 320], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dc922969590421c8a179591c578f87fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.076416015625], dtype='float16').reshape([1]),
            paddle.uniform([1, 32, 320, 320], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c3ccd5c71e6cce7f4581568bcaf17ef2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.06378173828125], dtype='float16').reshape([1]),
            paddle.uniform([1, 32, 320, 320], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cbb023c5976467f7cc04d616a4eff2ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.44140625], dtype='float16').reshape([1]),
            paddle.uniform([1, 32, 160, 160], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d68831e0dbdc1ae4cac9d28b30540247(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.48828125], dtype='float16').reshape([1]),
            paddle.uniform([1, 48, 160, 160], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0184adf594f533344c1bfe2ad4658960(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.375], dtype='float16').reshape([1]),
            paddle.uniform([1, 48, 160, 160], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_011c280062db51ecce82a846eceefbf2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.388916015625], dtype='float16').reshape([1]),
            paddle.uniform([1, 48, 160, 160], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_13da236ec84f3281d5e3c2b12c516003(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1700439453125], dtype='float16').reshape([1]),
            paddle.uniform([1, 48, 160, 160], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9f887812bdc2b595e7e6e0d89748064d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.393310546875], dtype='float16').reshape([1]),
            paddle.uniform([1, 48, 160, 160], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_73432a7df2178eb1cb1175fe75ed44ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.42626953125], dtype='float16').reshape([1]),
            paddle.uniform([1, 48, 160, 160], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_86008a3b22d82f9e34b91e824078ac24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.30810546875], dtype='float16').reshape([1]),
            paddle.uniform([1, 48, 80, 80], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c801c23d830c79d6ece1dceabb3f23b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.33984375], dtype='float16').reshape([1]),
            paddle.uniform([1, 96, 80, 80], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3d5b82aa1949cd9ebd057dfdd95154f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.30517578125], dtype='float16').reshape([1]),
            paddle.uniform([1, 96, 80, 80], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1a90f800d66e68bda3268fc4c8070523(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.043670654296875], dtype='float16').reshape([1]),
            paddle.uniform([1, 96, 80, 80], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c6c17c87385f14406d8f8a54032ae6e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3349609375], dtype='float16').reshape([1]),
            paddle.uniform([1, 96, 80, 80], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_43b9bebc9c1da27287b755011516f415(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2587890625], dtype='float16').reshape([1]),
            paddle.uniform([1, 96, 80, 80], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d0e137f89ad46c3c9f2fd95647b25790(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.14208984375], dtype='float16').reshape([1]),
            paddle.uniform([1, 96, 80, 80], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_73f327c5f620eafa7e0780e7fffc85d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0240478515625], dtype='float16').reshape([1]),
            paddle.uniform([1, 96, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_95e4e0e9dd054c64cf9ef7024c62dcbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.146484375], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_291dc1dd026b56a9a6eede034c0c45a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1553955078125], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_028c84d251e39ad386f512989370efba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.056915283203125], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9a13135ce7fc4321e94b2af847027382(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.07470703125], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5f8c585ac26d45230216a4ff449e74b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.429443359375], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_63ee90d5ab31c06380d76d252e2d167c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.036529541015625], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e819e5d60c5b126cfd8b562ef595754a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.09844970703125], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b6b1306c3a766a624c63fec94b81b111(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.348876953125], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_00c86d8d86b8b3110edfd245770f7ee6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.032470703125], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_30755e92203494ed9a779ffcc3be96fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2406005859375], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cbb471e9fd3eb1a779122cb667a90c2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.32177734375], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3d88dd6afadac7d5c65d372f61c8bacd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.34619140625], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d1378f9a93727c59ac7fe71cf9e8d283(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.434326171875], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fc3c680105506abcd385d9eb6ee87896(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.038543701171875], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_85a75499b5616133c5e1f9c50431a026(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.39453125], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c845204823badce68d869610c3337e73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0390625], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2439985f52dfd78c4ec128e317d6c41f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4619140625], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_85f409cf65b7f9a188842399ba3bd3ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.049224853515625], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_26426ebb0a33a4eacd1a168756e23a08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.387939453125], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 20, 20], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9362101e889f51500beb66c5c3c2cdb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 20, 20], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_df6939c2d24245dec6196072e4dbb13e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.06439208984375], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0e2ede167cdd1f093bb1feb5e9ba66d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.492431640625], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_30fcfa350c1c75bfadfa21b1651f157e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.29638671875], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8b32a2101f7ac6af1c789b886114f9e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10614013671875], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f16cf2e8d91baf0cbd2bfcf8201d5ec5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 20, 20], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_cb1eec4f05bd5c78060c302a50729b03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1583251953125], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6e6224d7cdf71aa7187d046cf21069fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1260986328125], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_93c90479bb84a03233dde4c38921f50c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.259765625], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b7c7b0ba655cbb84fb6d7ddaf9bb43a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.158447265625], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3d38a13e0c0b99ea16de2e9e9787dca2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.12200927734375], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8b218065754506bcd099f05c8cfc609d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1407470703125], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fee667778ca5254d8f4a2928147d8f1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.06878662109375], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c2471d0ed006e7a14d299272cdd67f23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.056671142578125], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9dc3cdaeab2947a36ecef70774211028(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.01555633544921875], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_699c20810ff2850a8266357c78f4c655(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2fa1abaf43c343dc65ff7d5f89404b5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.429931640625], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_da3cc893466494942ece16ffcba9b608(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 20, 20], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_f5402eb5b14d0d77435fe153b75f95ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 40, 40], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_d402b01c2b77583cac86dade30ba8593(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 80, 80], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_abfe3ba9f7e426aaa6daefbb1ba300fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 160, 160], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_884ab5c5723ba1e9e5350c7148c5781e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 20, 20], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float16').reshape([1, 24, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2d17c4229e84de488b5a5a1ce46dc982(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 40, 40], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float16').reshape([1, 24, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ae922f5a885ff121460d9f25517cb2a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 80, 80], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float16').reshape([1, 24, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_09738701c0d1a3cd72538e25f80b5169(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 160, 160], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float16').reshape([1, 24, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_71869dc44f59e06a7d57a9fe0e8e8736(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a8d361227507de65ad8728339b103d3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71869dc44f59e06a7d57a9fe0e8e8736
    def get_inputs(self):
        return [
            paddle.uniform([32, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ccfc8370fdb784b865dcb0e6516e3a55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71869dc44f59e06a7d57a9fe0e8e8736
    def get_inputs(self):
        return [
            paddle.uniform([64, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_40b801dafbaec481e73e1978fb153109(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71869dc44f59e06a7d57a9fe0e8e8736
    def get_inputs(self):
        return [
            paddle.uniform([160, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_791280c2d4a9b5fc3c294d8b073d02ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71869dc44f59e06a7d57a9fe0e8e8736
    def get_inputs(self):
        return [
            paddle.uniform([256, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c74f93bdfbd08a82b2d9431f5462d3f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 80, 80], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6cef5c1f8325213c14621d5daf63ad2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 80, 80], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_7efe3edda05cfcc1f852e9d7efc9bcab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 40, 40], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f3df0aef0861d9d7bb3a12dd935efb39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 40, 40], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9af8649e43360dda359ca377e3531ed4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 20, 20], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5e97fcfda19165e522490628363f132a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7ab34b8db1c80002ff78cc90fb83943
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 20, 20], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_140e9ba55f860461bacc65487af9993a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_779dd1a1352b01918a83b3635ab6a202
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 40, 40], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_25f2a1325a74f9bd098e25878a756e47(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5d090e94bffc17f3c192d50dd7badfc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25f2a1325a74f9bd098e25878a756e47
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor(160, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_00aced8cc2d170fda61b68065124b0f4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_55472311ff5b9b4d56ef7851d71c14cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00aced8cc2d170fda61b68065124b0f4
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 128], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 128, 128, 128], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c97b66b660494e0d07f6a947f1b5b26f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00aced8cc2d170fda61b68065124b0f4
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 32], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 128, 32, 32], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_37d7483dad4edab16b886ad99ba7c8a1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 26, 512], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 26, 512], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a47546358e83db5baa6ef39f9b431d31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37d7483dad4edab16b886ad99ba7c8a1
    def get_inputs(self):
        return [
            paddle.uniform([1, 26, 512], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 26, 512], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_760af7e536f24fc22f2cf0c21802ae25(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 256, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2abf0a71eecb517188f6b1b48109ddcc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_760af7e536f24fc22f2cf0c21802ae25
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_ead7f2bcf4bf6fd9076f77e637a38c44(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 512, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e5f80cc4885363d54f2570d929ce3815(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ead7f2bcf4bf6fd9076f77e637a38c44
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 16, 16], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_1bb44fab1351388b44d6a48973f78864(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f791a54c03c05f3c7a9cea480f911587(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bb44fab1351388b44d6a48973f78864
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 128, 128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c1f05880d6e4a657e10fe027761d833f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bb44fab1351388b44d6a48973f78864
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 128, 32, 32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_492530f7fe55278055293cd3b60ea833(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25f2a1325a74f9bd098e25878a756e47
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor(240, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a2fedc85369ffb9ecc024553605154e1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 40, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 40, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e7c89e0bcae01f0c21ac9e8bc6f1c79a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2fedc85369ffb9ecc024553605154e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 120, 120], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_959b775eab74bbd70138b39517e94f07(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 64, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_828613f2fe98e97a241b781a087bfe0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_959b775eab74bbd70138b39517e94f07
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 120, 120], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_9235e831308f4e9c7ba9e3b8d4a4407c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 240, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dd14de84b46babe6d569584c5d8bd12b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9235e831308f4e9c7ba9e3b8d4a4407c
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 60, 60], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_cd264045cb0e22f7c3d28013ff3477dd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 336, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 336, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a39e2cc5fbb8852bc9adb0601b51f73e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd264045cb0e22f7c3d28013ff3477dd
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 60, 60], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_800f6e3131ae955b279c9bad516c054d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd264045cb0e22f7c3d28013ff3477dd
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 30, 30], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_701e5b72f8d7a1806675c65bd1524ad3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 480, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 480, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b1ad3defb092fcf63227cbcb31197288(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_701e5b72f8d7a1806675c65bd1524ad3
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 30, 30], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_d936e8b173b8c9ec0cd3bbb4d8dbd821(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 384, 20, 20], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 384, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c3be9b80d6bb55864cf67f00d8a0bd89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d936e8b173b8c9ec0cd3bbb4d8dbd821
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_c7382078ac8d360ffed77d4bf31b7117(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, 40, 40], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 192, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9b0a1e3332e7e209f0ec5bcdfbcf3d8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7382078ac8d360ffed77d4bf31b7117
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_02ffeef33d8882b97766a6525b32d97c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 80, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a04fc8ae5d8a422d26222f1cca5683e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02ffeef33d8882b97766a6525b32d97c
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 80, 80], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_6f90e312cf9d366a340b2dc654a849b9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[1, 3, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1b91ee978bcb1f514d2d06ad65d46913(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f90e312cf9d366a340b2dc654a849b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 800, 1344], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([[[[0.071533203125]], [[0.231689453125]], [[0.33837890625]]]], dtype='float16').reshape([1, 3, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4e2810397177d4ea8e2931404b433b6c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e234f68c902a8afe6e0bc36f6a72b0a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e2810397177d4ea8e2931404b433b6c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 100, 168], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2371826171875], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a00591c585f692d261054447ef195610(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e2810397177d4ea8e2931404b433b6c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 50, 84], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.09674072265625], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4a8a1844f2a233d7a04e8f26e55f3784(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e2810397177d4ea8e2931404b433b6c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 25, 42], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1575927734375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8ed0f8667b72169b3768dde87e9484d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e2810397177d4ea8e2931404b433b6c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 13, 21], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.08050537109375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0fe6515661197da23bf52111420aa545(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e2810397177d4ea8e2931404b433b6c
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 7, 11], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.310302734375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4f92668774df2bc578f506b779a2f3f8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 80], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_54c3695fdf1591bd9b889040bbf8eb4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f92668774df2bc578f506b779a2f3f8
    def get_inputs(self):
        return [
            paddle.uniform([1, 16800, 80], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 16800, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1dffbe120feed5013aee9d26868e7586(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f92668774df2bc578f506b779a2f3f8
    def get_inputs(self):
        return [
            paddle.uniform([1, 4200, 80], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 4200, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_46c124393cf6b8fca8e1429b202788e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f92668774df2bc578f506b779a2f3f8
    def get_inputs(self):
        return [
            paddle.uniform([1, 1050, 80], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1050, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_48c43f1a79086261d22dcab833371ce4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f92668774df2bc578f506b779a2f3f8
    def get_inputs(self):
        return [
            paddle.uniform([1, 273, 80], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 273, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_75da0eb329af6316e5013866385109dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f92668774df2bc578f506b779a2f3f8
    def get_inputs(self):
        return [
            paddle.uniform([1, 77, 80], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 77, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ede4361d29a978ccb9ad4049552cccfa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 72, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 72, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0e0ead2ae08f1e647694baa60ad028be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ede4361d29a978ccb9ad4049552cccfa
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 128, 64], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_919327c02b63bc8d9eacb7b041a73776(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 120, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 120, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_98e3cb81d51983e55816deea45d136c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_919327c02b63bc8d9eacb7b041a73776
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 128, 64], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_0c8b0b4c329c20902f2f56b6fa72d605(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 480, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 480, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3f3b93a33b6a833eff86d2813cd14291(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8b0b4c329c20902f2f56b6fa72d605
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 64, 32], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_43587c3cac3490b94b4a71fd9faa023a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 672, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 672, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e357de6fa49048c0a86ac45b752f4452(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43587c3cac3490b94b4a71fd9faa023a
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 64, 32], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_8e41719a9843c7313c048f7c04f1904a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43587c3cac3490b94b4a71fd9faa023a
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 32, 16], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_010e4620f346a3cdff2fb2523aaef9a7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 960, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 960, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_21fd010cd5f895cb4495b847c004ad65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_010e4620f346a3cdff2fb2523aaef9a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 32, 16], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_1810f49fff5bb6ead7056a3f5116447d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2, 1, 9, 112, 112], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 2, 16, 9, 112, 112], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_12edb57a58f4a77645bd26bcc5f026eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1810f49fff5bb6ead7056a3f5116447d
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 1, 9, 112, 112], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 2, 16, 9, 112, 112], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_bae77b97a71a035862e764902d60668f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 1, 49, 56, 56], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 4, 16, 49, 56, 56], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bf1a15d1dcddca08c60d7c6e70e523f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bae77b97a71a035862e764902d60668f
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1, 49, 56, 56], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 4, 16, 49, 56, 56], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2da0b923eb0c401680d11cff8b4b96ac(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, 1, 49, 28, 28], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 8, 16, 49, 28, 28], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6236be198a0adb42cf8cf90d1dc84b44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2da0b923eb0c401680d11cff8b4b96ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1, 49, 28, 28], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 8, 16, 49, 28, 28], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2b8c3b899c491180d47ca19d18b508ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 1, 49, 14, 14], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 16, 16, 49, 14, 14], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8a77415974786650cf02f83d8bfcf780(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b8c3b899c491180d47ca19d18b508ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 1, 49, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 16, 16, 49, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0b791198af8d79c73645c257ce72043b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 1, 49, 7, 7], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 32, 16, 49, 7, 7], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d6dba065bbd2be78ed3194559e069e62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0b791198af8d79c73645c257ce72043b
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 49, 7, 7], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 32, 16, 49, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ad96b4ff0519273f7db2993654355938(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 16, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_59f8860425d3d259c39bac3c145ff98d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad96b4ff0519273f7db2993654355938
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4784398078918457], dtype='float32').reshape([1]),
            paddle.uniform([1, 16, 480, 480], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ed482369b5096363e906a25905ccbdf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad96b4ff0519273f7db2993654355938
    def get_inputs(self):
        return [
            paddle.to_tensor([0.14870469272136688], dtype='float32').reshape([1]),
            paddle.uniform([1, 16, 480, 480], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_25a9b384a8df1f2393ddef8fd3a80bab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b03710d1197ed88ad11e910b6ab47e10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25a9b384a8df1f2393ddef8fd3a80bab
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1273345649242401], dtype='float32').reshape([1]),
            paddle.uniform([1, 32, 480, 480], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3018ed63b9a4bcd9742dcd20ca31bd3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25a9b384a8df1f2393ddef8fd3a80bab
    def get_inputs(self):
        return [
            paddle.to_tensor([0.011404834687709808], dtype='float32').reshape([1]),
            paddle.uniform([1, 32, 480, 480], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_73e6c67ce7f5570711f6642541e70804(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25a9b384a8df1f2393ddef8fd3a80bab
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4188338816165924], dtype='float32').reshape([1]),
            paddle.uniform([1, 32, 240, 240], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f6cbc81f9d908468332d470608147155(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_97ebce9e3d2c82e94078819d293a583b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6cbc81f9d908468332d470608147155
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4657136797904968], dtype='float32').reshape([1]),
            paddle.uniform([1, 48, 240, 240], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c86a96e018219daf16bc74fe826714a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6cbc81f9d908468332d470608147155
    def get_inputs(self):
        return [
            paddle.to_tensor([0.13037486374378204], dtype='float32').reshape([1]),
            paddle.uniform([1, 48, 240, 240], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e14661917ecd1c0832d09ce034c98035(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6cbc81f9d908468332d470608147155
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1505524218082428], dtype='float32').reshape([1]),
            paddle.uniform([1, 48, 240, 240], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6cafd873bd40be1d3a35c65467e7b100(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6cbc81f9d908468332d470608147155
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0886026993393898], dtype='float32').reshape([1]),
            paddle.uniform([1, 48, 240, 240], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_490a25bd40ca81f5a0700e20929e23b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6cbc81f9d908468332d470608147155
    def get_inputs(self):
        return [
            paddle.to_tensor([0.17881451547145844], dtype='float32').reshape([1]),
            paddle.uniform([1, 48, 240, 240], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a86c8ec370cb04f0ea2d851b4f7dddc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6cbc81f9d908468332d470608147155
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4926683306694031], dtype='float32').reshape([1]),
            paddle.uniform([1, 48, 240, 240], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a0aa2af4a25d92b144a5797a0906ff45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6cbc81f9d908468332d470608147155
    def get_inputs(self):
        return [
            paddle.to_tensor([0.329205721616745], dtype='float32').reshape([1]),
            paddle.uniform([1, 48, 120, 120], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_29a3dcdcc57b20a7a8622b91fac35f48(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0c9bb068af22bb012f4c56b06631799a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29a3dcdcc57b20a7a8622b91fac35f48
    def get_inputs(self):
        return [
            paddle.to_tensor([0.43924281001091003], dtype='float32').reshape([1]),
            paddle.uniform([1, 96, 120, 120], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1133230c71485068fdf9c443c6280d79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29a3dcdcc57b20a7a8622b91fac35f48
    def get_inputs(self):
        return [
            paddle.to_tensor([0.05343525856733322], dtype='float32').reshape([1]),
            paddle.uniform([1, 96, 120, 120], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b0ecc9c58046a457de0b3cfa4abaf363(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29a3dcdcc57b20a7a8622b91fac35f48
    def get_inputs(self):
        return [
            paddle.to_tensor([0.41747212409973145], dtype='float32').reshape([1]),
            paddle.uniform([1, 96, 120, 120], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e71a5dbbee4feeb174eaa0a71ad3c4f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29a3dcdcc57b20a7a8622b91fac35f48
    def get_inputs(self):
        return [
            paddle.to_tensor([0.13127554953098297], dtype='float32').reshape([1]),
            paddle.uniform([1, 96, 120, 120], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d89a7e4bfcecd059fdfe75440abada86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29a3dcdcc57b20a7a8622b91fac35f48
    def get_inputs(self):
        return [
            paddle.to_tensor([0.49068695306777954], dtype='float32').reshape([1]),
            paddle.uniform([1, 96, 120, 120], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_63d24c2d46e60d037a32f9e53acb04e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29a3dcdcc57b20a7a8622b91fac35f48
    def get_inputs(self):
        return [
            paddle.to_tensor([0.11958789825439453], dtype='float32').reshape([1]),
            paddle.uniform([1, 96, 120, 120], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0dc7c07c8c6685bf03ed7ae62410edbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29a3dcdcc57b20a7a8622b91fac35f48
    def get_inputs(self):
        return [
            paddle.to_tensor([0.34703755378723145], dtype='float32').reshape([1]),
            paddle.uniform([1, 96, 60, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5a28e6e94ec21319929921665e8cfc9d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c9a6f5f61d2a58d1620ae788feb1b088(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a28e6e94ec21319929921665e8cfc9d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4826791286468506], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f44804cbf7036017a442547b2cb7ec86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a28e6e94ec21319929921665e8cfc9d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4372650384902954], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3477e9aff288b90321e8f0fd499cacc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a28e6e94ec21319929921665e8cfc9d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.13426004350185394], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_88e031d7941a6f4eb4b9fcd40c121616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a28e6e94ec21319929921665e8cfc9d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.20082494616508484], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a0ac8ec1e439bd2510e40221fc6d340a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a28e6e94ec21319929921665e8cfc9d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1471903771162033], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f84a759da1c2a70b5cdbe4def1d89407(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a28e6e94ec21319929921665e8cfc9d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.19031450152397156], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_75370c8e922df0146722b158948e6e13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a28e6e94ec21319929921665e8cfc9d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.23371702432632446], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d9bc77bd28411efd975dcc8cc2a4768d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a28e6e94ec21319929921665e8cfc9d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.460300475358963], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7e4c8c433c1295e337353c557f0f8277(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a28e6e94ec21319929921665e8cfc9d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.14110355079174042], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_be396c724242b16b1e7575d59dd186dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a28e6e94ec21319929921665e8cfc9d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10759323835372925], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4813e7d52688a3364602b25c393bb1a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a28e6e94ec21319929921665e8cfc9d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.13945358991622925], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_69b5f962bb65621ca4a84a3509f79fff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a28e6e94ec21319929921665e8cfc9d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1550578624010086], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_571bb32a91394972c1e8e311ebd9ff01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a28e6e94ec21319929921665e8cfc9d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.18472763895988464], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d80bc7bc8eaa7ef9babf6fe9e7be8b29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a28e6e94ec21319929921665e8cfc9d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.48091572523117065], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9c6bb6ddb15453c88fe8e84767849744(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a28e6e94ec21319929921665e8cfc9d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.40070024132728577], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_739c4ec3dd38679a9a0d2f2558c7b922(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a28e6e94ec21319929921665e8cfc9d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.03564074635505676], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b8d6cf64b609f4d380cb7b1e43f29024(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a28e6e94ec21319929921665e8cfc9d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.14680147171020508], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d6631567100adb40924b401e5403c96c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a28e6e94ec21319929921665e8cfc9d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.46726563572883606], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_00a819beb6978007d0c485f1ad989b2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a28e6e94ec21319929921665e8cfc9d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.09462948143482208], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 30, 30], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_de55f9ce9eade27536df339da542b103(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 192, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6c83b7641c65b30fbbbf40b8e0160264(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de55f9ce9eade27536df339da542b103
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 30, 30], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_22e2bd3a5a9602d84b8fae1fdff2989e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 384, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f83acdcd2f67af63e37786b02df14ce2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22e2bd3a5a9602d84b8fae1fdff2989e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.08917675912380219], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d5c946bb5eb12988b0724c4a98523445(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22e2bd3a5a9602d84b8fae1fdff2989e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10074345767498016], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_edf8d61e952fde598b313d7e3b68d178(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22e2bd3a5a9602d84b8fae1fdff2989e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.34102490544319153], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_11d4b8ade1bc9c8e135d801bc66aec4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22e2bd3a5a9602d84b8fae1fdff2989e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3246842622756958], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_7067e826a5fca59676079df4bf849961(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 384, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 384, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_45a6b821572bb8c3528a7de66d9c4607(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7067e826a5fca59676079df4bf849961
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 30, 30], dtype='float32', min=0, max=0.5),
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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4b6c5c9c55e7f33bb9081d46bf5477af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22e2bd3a5a9602d84b8fae1fdff2989e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3248758912086487], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_361fdce0d5d5d6ee41e2d798b0d62e48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22e2bd3a5a9602d84b8fae1fdff2989e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.20371150970458984], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e20def063f2357ce69e1f87ca6cd8588(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22e2bd3a5a9602d84b8fae1fdff2989e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.028545115143060684], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_193ed9f06ebab2b8d0ceebbffaf069ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22e2bd3a5a9602d84b8fae1fdff2989e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.26297762989997864], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d45178cc9905c5c7ed1d3381bd664763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22e2bd3a5a9602d84b8fae1fdff2989e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.06808995455503464], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_49a71dc8b46bc65389ba1aa496d6aba8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22e2bd3a5a9602d84b8fae1fdff2989e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1499202996492386], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d86dfcbb0270fab092c42662bea88370(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22e2bd3a5a9602d84b8fae1fdff2989e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.044834017753601074], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4b10312eb34400295782eb3788619858(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22e2bd3a5a9602d84b8fae1fdff2989e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.21141599118709564], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0637617535dd6a00fa38c6d66fc48182(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22e2bd3a5a9602d84b8fae1fdff2989e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.31889456510543823], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cfe985f833ed2bd01151fbe84f5f7a3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22e2bd3a5a9602d84b8fae1fdff2989e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.24018998444080353], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5c8ef0676d9d343787fed8c6d03fea09(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b8825557bdeb34970c2ed75f19baa8f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c8ef0676d9d343787fed8c6d03fea09
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 30, 30], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_0791833020b97f737bae4b0009040757(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c8ef0676d9d343787fed8c6d03fea09
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 60, 60], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_3564fadb97e516c5c6d02dd368ac31b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c8ef0676d9d343787fed8c6d03fea09
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 120, 120], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ab8c296ea14205472d2009e16d7c8cb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c8ef0676d9d343787fed8c6d03fea09
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 240, 240], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_7c8c92f07f7903913d2cbf14b8c2e5d5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 24, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 24, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_aa269764187836249dc7aa2c935a70b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c8c92f07f7903913d2cbf14b8c2e5d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_839b57efe45632861f38989ef73a0c35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c8c92f07f7903913d2cbf14b8c2e5d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b091269af7e3108305631c91a252775f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c8c92f07f7903913d2cbf14b8c2e5d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 120, 120], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_04c7f96a21a9aeef09714b3098a9574a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c8c92f07f7903913d2cbf14b8c2e5d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 240, 240], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_82a852fcc97a004d3cb9813578e0ed9a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[32, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 32, 56, 56], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8f81522431915ddd0d9ad38f74fb798c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_82a852fcc97a004d3cb9813578e0ed9a
    def get_inputs(self):
        return [
            paddle.uniform([32, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 32, 56, 56], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_288213d99e15cf317afe869ce574f409(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[64, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 64, 28, 28], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_49e6a361823a6b5a0a90622d6a91f76d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_288213d99e15cf317afe869ce574f409
    def get_inputs(self):
        return [
            paddle.uniform([64, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 64, 28, 28], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_21b50ad142813e7faab6bc72e38a7463(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[160, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 160, 14, 14], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_05365e24a34dc92f8d1e14359040e2bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21b50ad142813e7faab6bc72e38a7463
    def get_inputs(self):
        return [
            paddle.uniform([160, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 160, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_46eab726a00d09c2056492855eac95df(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[256, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 256, 7, 7], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d2b7faa2e2b4f49bb942929edc14f4df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46eab726a00d09c2056492855eac95df
    def get_inputs(self):
        return [
            paddle.uniform([256, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 256, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0a2dd97b71096979f8c6e38a1c399216(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 28, 28], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 32, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6e2f35eb0fb756d774e7eeb276c631c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a2dd97b71096979f8c6e38a1c399216
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 28, 28], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_d39e29a610cf633f23f1c430e89c2205(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, 14, 14], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 48, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_89383af5be32365d5ae4e414f9c578ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d39e29a610cf633f23f1c430e89c2205
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 14, 14], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_d908cc6be4c79876becc99e6864efb41(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 14, 14], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 64, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b12c0bb97901d09bffd2a85f9938dfb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d908cc6be4c79876becc99e6864efb41
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 14, 14], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_966c1886a84c0049298a3cf269efd29c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 14, 14], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 128, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_de3c7f9ff297f0793e6614facc900a00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_966c1886a84c0049298a3cf269efd29c
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 14, 14], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_dafb05762bd77f0edb4abf38d0b912a4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 7, 7], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 128, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fe78125613d802b18042705877e5fe90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dafb05762bd77f0edb4abf38d0b912a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 7, 7], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_d0245ee304ff72ced72bd8d7e76aa3af(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 7, 7], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 256, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b0da06548338f623cf65c1ed3c486f84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d0245ee304ff72ced72bd8d7e76aa3af
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 7, 7], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_a25c7c6aeba2080e4c338ea05a69adf1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 16, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b9fa3a949df9afd65737a6b1a54cbbd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a25c7c6aeba2080e4c338ea05a69adf1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.319091796875], dtype='float16').reshape([1]),
            paddle.uniform([1, 16, 480, 480], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_777df36f368c3e9021c7f72c5be789b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a25c7c6aeba2080e4c338ea05a69adf1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.337890625], dtype='float16').reshape([1]),
            paddle.uniform([1, 16, 480, 480], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d39d116a6d93012a5795633b24b57810(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f70381b9eb6d94e5245ff937e225bd26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d39d116a6d93012a5795633b24b57810
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0692138671875], dtype='float16').reshape([1]),
            paddle.uniform([1, 32, 480, 480], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e2ed4edf0f7a5e7aa81d4ce0ce53242f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d39d116a6d93012a5795633b24b57810
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2271728515625], dtype='float16').reshape([1]),
            paddle.uniform([1, 32, 480, 480], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_51b139f8e2193a526a80d5f2c39e1f94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d39d116a6d93012a5795633b24b57810
    def get_inputs(self):
        return [
            paddle.to_tensor([0.459716796875], dtype='float16').reshape([1]),
            paddle.uniform([1, 32, 240, 240], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_fe3fca136e37242dc43c1fbccc2fde49(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c89bc806d4904bd591f91b13ba33699b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe3fca136e37242dc43c1fbccc2fde49
    def get_inputs(self):
        return [
            paddle.to_tensor([0.267578125], dtype='float16').reshape([1]),
            paddle.uniform([1, 48, 240, 240], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_34a25f1c23ec0b4b0359c6763523f859(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe3fca136e37242dc43c1fbccc2fde49
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0087127685546875], dtype='float16').reshape([1]),
            paddle.uniform([1, 48, 240, 240], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_de2ccee6285cb08f8dfe3cf0e5e4699e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe3fca136e37242dc43c1fbccc2fde49
    def get_inputs(self):
        return [
            paddle.to_tensor([0.369140625], dtype='float16').reshape([1]),
            paddle.uniform([1, 48, 240, 240], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dde341e11793e42b719a83cbdd6efa45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe3fca136e37242dc43c1fbccc2fde49
    def get_inputs(self):
        return [
            paddle.to_tensor([0.007537841796875], dtype='float16').reshape([1]),
            paddle.uniform([1, 48, 240, 240], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ee034ee3898a7dd54f9d9d17ff68f3c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe3fca136e37242dc43c1fbccc2fde49
    def get_inputs(self):
        return [
            paddle.to_tensor([0.078125], dtype='float16').reshape([1]),
            paddle.uniform([1, 48, 240, 240], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b2eaf8ee31589b08925411cf84bb3fb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe3fca136e37242dc43c1fbccc2fde49
    def get_inputs(self):
        return [
            paddle.to_tensor([0.128662109375], dtype='float16').reshape([1]),
            paddle.uniform([1, 48, 240, 240], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bc1f98d9b33af823d22d5fb9ca0bda4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe3fca136e37242dc43c1fbccc2fde49
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1458740234375], dtype='float16').reshape([1]),
            paddle.uniform([1, 48, 120, 120], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c45923d4d5fc586b60fc1bb097fed418(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0527b087e69508e72e4de65672c80bbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c45923d4d5fc586b60fc1bb097fed418
    def get_inputs(self):
        return [
            paddle.to_tensor([0.267578125], dtype='float16').reshape([1]),
            paddle.uniform([1, 96, 120, 120], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e34369eebd47dd95acc17db7831bf102(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c45923d4d5fc586b60fc1bb097fed418
    def get_inputs(self):
        return [
            paddle.to_tensor([0.395263671875], dtype='float16').reshape([1]),
            paddle.uniform([1, 96, 120, 120], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4ca9cbd16aa05970eb7c1f18b0246e69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c45923d4d5fc586b60fc1bb097fed418
    def get_inputs(self):
        return [
            paddle.to_tensor([0.191650390625], dtype='float16').reshape([1]),
            paddle.uniform([1, 96, 120, 120], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cbfe99ea7606f55adbf57502e6c7a505(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c45923d4d5fc586b60fc1bb097fed418
    def get_inputs(self):
        return [
            paddle.to_tensor([0.19287109375], dtype='float16').reshape([1]),
            paddle.uniform([1, 96, 120, 120], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_722e69b2ce014fd8ff8330cf06a9571b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c45923d4d5fc586b60fc1bb097fed418
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4609375], dtype='float16').reshape([1]),
            paddle.uniform([1, 96, 120, 120], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_91c1a90325e556864007c40c5dbec09d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c45923d4d5fc586b60fc1bb097fed418
    def get_inputs(self):
        return [
            paddle.to_tensor([0.187255859375], dtype='float16').reshape([1]),
            paddle.uniform([1, 96, 120, 120], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_97099a5ae67bbc09b5e3141b9aca1b9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c45923d4d5fc586b60fc1bb097fed418
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1800537109375], dtype='float16').reshape([1]),
            paddle.uniform([1, 96, 60, 60], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_689176cc2300dfd17d02a2f49a9cb30c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ef804e0f9d94d613a11e0b697072b6a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_689176cc2300dfd17d02a2f49a9cb30c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.43896484375], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2e3edfd3ec1539eafaccc824b3208c59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_689176cc2300dfd17d02a2f49a9cb30c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.156005859375], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e13a813b5780c8430cff387ece044da5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_689176cc2300dfd17d02a2f49a9cb30c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.48974609375], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_96597cbb9762e8d8c78818e3d4607329(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_689176cc2300dfd17d02a2f49a9cb30c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.451171875], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_17f610af6e7821a76ac56312a65c5dde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_689176cc2300dfd17d02a2f49a9cb30c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.384521484375], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c149fd1ccc001118f92a7311dfdb8029(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_689176cc2300dfd17d02a2f49a9cb30c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.352294921875], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f4a6926093af99863f2d0a1e52cb14ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_689176cc2300dfd17d02a2f49a9cb30c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.249267578125], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_deef0526f68a10a0493f6d8aec6049a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_689176cc2300dfd17d02a2f49a9cb30c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1373291015625], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_185feaa6c99d6694898757c09e7252ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_689176cc2300dfd17d02a2f49a9cb30c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.423828125], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_84f915b80fe9f220000dd16513888e90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_689176cc2300dfd17d02a2f49a9cb30c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.234619140625], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5fa3ee5038f39666b650b8382b15ef7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_689176cc2300dfd17d02a2f49a9cb30c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3466796875], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9be0144a645c98dcd49c195cee01d7c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_689176cc2300dfd17d02a2f49a9cb30c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.346435546875], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5912b67b52c154693aee6e9e3f386688(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_689176cc2300dfd17d02a2f49a9cb30c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3779296875], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9519346fc60298495998acea5f3f31fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_689176cc2300dfd17d02a2f49a9cb30c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.11480712890625], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_68c9efba8a55bbf0b8825e536cb8e9bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_689176cc2300dfd17d02a2f49a9cb30c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0653076171875], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0ed517103ec8e14926ff6ddf8405f2ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_689176cc2300dfd17d02a2f49a9cb30c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.357177734375], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9269dd4e9015b283bd6a42a8b38f9b35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_689176cc2300dfd17d02a2f49a9cb30c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3251953125], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c77c50eca4efdfa467fd096aaba4033a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_689176cc2300dfd17d02a2f49a9cb30c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.408203125], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 60, 60], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_919d3420caf7aefdeb2a67ed6074a2a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_689176cc2300dfd17d02a2f49a9cb30c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1578369140625], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 30, 30], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_63b159b310cc8c7c69a8fd046ebd1b5e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 192, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c3d77a5c1ea6b72183d4697d0f253622(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63b159b310cc8c7c69a8fd046ebd1b5e
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 30, 30], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_e0dd8f88cb736ffc47e4f6d2556dc562(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 384, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_81b88f09411a1e8c588a717d568b8fa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0dd8f88cb736ffc47e4f6d2556dc562
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0192108154296875], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7181712af4aada9dcfcf0948ea33c307(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0dd8f88cb736ffc47e4f6d2556dc562
    def get_inputs(self):
        return [
            paddle.to_tensor([0.03240966796875], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7aef4a4a323fc0b83e4ace25223fd596(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0dd8f88cb736ffc47e4f6d2556dc562
    def get_inputs(self):
        return [
            paddle.to_tensor([0.280029296875], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ea241e61fb1086ddb164bae328f173b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0dd8f88cb736ffc47e4f6d2556dc562
    def get_inputs(self):
        return [
            paddle.to_tensor([0.421875], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2935bfe3850f9f8ad92d3dc68a8ad941(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 384, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 384, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6987d810461781e069c5327f4dbf36c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2935bfe3850f9f8ad92d3dc68a8ad941
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 30, 30], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_2b1805d2565cfadcf93c99ab38d55120(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0dd8f88cb736ffc47e4f6d2556dc562
    def get_inputs(self):
        return [
            paddle.to_tensor([0.027099609375], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c16b737fedd9daf746649eea12137d6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0dd8f88cb736ffc47e4f6d2556dc562
    def get_inputs(self):
        return [
            paddle.to_tensor([0.461181640625], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_780c703d41d980531956aa7a422e0ff1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0dd8f88cb736ffc47e4f6d2556dc562
    def get_inputs(self):
        return [
            paddle.to_tensor([0.348388671875], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c5b4fc7e5660a52db374233e292f4871(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0dd8f88cb736ffc47e4f6d2556dc562
    def get_inputs(self):
        return [
            paddle.to_tensor([0.34423828125], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0510a3d4f920465120a7b48d48396d67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0dd8f88cb736ffc47e4f6d2556dc562
    def get_inputs(self):
        return [
            paddle.to_tensor([0.281494140625], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_573e6ae77bcb4eef33c2b1397cd84602(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0dd8f88cb736ffc47e4f6d2556dc562
    def get_inputs(self):
        return [
            paddle.to_tensor([0.09295654296875], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f945d2d0b509a696b3c777cae61cd3a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0dd8f88cb736ffc47e4f6d2556dc562
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2174072265625], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_444c0ed3cda1c1d3c3e2113d7ecedc10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0dd8f88cb736ffc47e4f6d2556dc562
    def get_inputs(self):
        return [
            paddle.to_tensor([0.22265625], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ce3e0cb411b27972ba983e3b8a5f2da3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0dd8f88cb736ffc47e4f6d2556dc562
    def get_inputs(self):
        return [
            paddle.to_tensor([0.476806640625], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a25029e7cc85f5a61d9f240befc7c162(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0dd8f88cb736ffc47e4f6d2556dc562
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3193359375], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 30, 30], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_32276cd30b45b13643f12def8f1074f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 96, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_977cd3be7d2995959532a735e1ecf1fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32276cd30b45b13643f12def8f1074f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 30, 30], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_e3c5c632bca54534da8c0fc4c4c5e7db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32276cd30b45b13643f12def8f1074f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 60, 60], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_67178c0f1aa32f9f6651ab2075c9bca7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32276cd30b45b13643f12def8f1074f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 120, 120], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_74cd7bca9e02e57a5a0d6e67e9d4e6a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32276cd30b45b13643f12def8f1074f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 240, 240], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_6c54c838f0e6717b6c3138519561c831(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 24, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 24, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a4e9b8773a3c2511ed8e0a23717151f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c54c838f0e6717b6c3138519561c831
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 30, 30], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float16').reshape([1, 24, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c72d893cae4c880ea6f2ce469c02da7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c54c838f0e6717b6c3138519561c831
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 60, 60], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float16').reshape([1, 24, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5efd1af02acc2b2cbaf3a59ba4bf4e8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c54c838f0e6717b6c3138519561c831
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 120, 120], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float16').reshape([1, 24, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c8df9a9464af56075cf841bf259c72bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c54c838f0e6717b6c3138519561c831
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 240, 240], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float16').reshape([1, 24, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4281d04b2d144ce9f96cdcf7f3238d9b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 72, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 72, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e96b5c937a9f54c8cdac56e57531b211(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4281d04b2d144ce9f96cdcf7f3238d9b
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 128, 64], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_205ad02c3ed973b364bb720146e27b6f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 120, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 120, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b60c28be1f82ad6e16d124717fddcede(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_205ad02c3ed973b364bb720146e27b6f
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 128, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_06ab3c1d3f725eb1a4fcae51fb7bcbad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_701e5b72f8d7a1806675c65bd1524ad3
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 64, 32], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_ecd527761dd44ac3c53d90244813c43c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 672, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 672, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_afb879bdc9634ddb55e2d368809a6242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ecd527761dd44ac3c53d90244813c43c
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 64, 32], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ed6440cea6bf59b03b30cdd24ac94a1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ecd527761dd44ac3c53d90244813c43c
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 32, 16], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_fa0bea37b7ee7dadc66f4e0a191a73cf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 960, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 960, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0d60e02ff7fbd3f52fc4eb20175ac5f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fa0bea37b7ee7dadc66f4e0a191a73cf
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 32, 16], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_7b1113b634be2cff1e120522333cb379(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, 1, 26], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 512, 1, 26], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_84b39d6eb466ca743abef4061a492ffd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b1113b634be2cff1e120522333cb379
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 26], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 26], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4be3ff3df328439d8e6235304e0f2336(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 196, 320], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0aff6b022478e4c6cccb32f5a545e162(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4be3ff3df328439d8e6235304e0f2336
    def get_inputs(self):
        return [
            paddle.uniform([320], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 196, 320], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a77dd655a1ea50ad7e7d3105cb310c1f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 49, 512], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a52755a8b83eb694bccdbc08d12bac4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a77dd655a1ea50ad7e7d3105cb310c1f
    def get_inputs(self):
        return [
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 49, 512], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6bada91df609be0bcb0b21ce65f1dcff(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, 1, 26], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 512, 1, 26], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6923dbf05249a0393ae7bfce159dd61f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bada91df609be0bcb0b21ce65f1dcff
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 26], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 512, 1, 26], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_986be6b479d0fbd9c1c1b95871b04b42(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 40, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 40, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cecec6af69e87b7153692639be630e14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986be6b479d0fbd9c1c1b95871b04b42
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 120, 120], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_b34bb4ef69193a81fd98f1f0dd09f8c8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 64, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bf18533bfb80ed5c3839645959b7f1e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b34bb4ef69193a81fd98f1f0dd09f8c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 120, 120], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_a2bad6d29dc72b5bff76b4c6e1a5034e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 240, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_61325b0a294b8cc28cf8ffab411ff496(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2bad6d29dc72b5bff76b4c6e1a5034e
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 60, 60], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_af1528c59b4bcac06a62b37b374f0b10(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 336, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 336, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_00a5ca6bbc58e155c71184df46891c91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af1528c59b4bcac06a62b37b374f0b10
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 60, 60], dtype='float16', min=0, max=0.5),
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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5ed39ef59f77d2c92480860397420e84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af1528c59b4bcac06a62b37b374f0b10
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 30, 30], dtype='float16', min=0, max=0.5),
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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8ba4245de07845ae1d9ca5cef79c349a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8b0b4c329c20902f2f56b6fa72d605
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 30, 30], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_1544c01250d326f6bcd71121281567da(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5846129686ae759cb664898664301662(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1544c01250d326f6bcd71121281567da
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 800, 1344], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.07380150258541107]], [[0.3107784688472748]], [[0.39664655923843384]]]], dtype='float32').reshape([1, 3, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_564531cf15b0403b21d6702acac85d15(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0cd1828b298ac50c6cfb6b66d7258bae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_564531cf15b0403b21d6702acac85d15
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 100, 168], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.44416195154190063], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e67b6073d4f5540715a9e81e910243b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_564531cf15b0403b21d6702acac85d15
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 50, 84], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.41823989152908325], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9628b95ce041cf799b515b32c6100ff6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_564531cf15b0403b21d6702acac85d15
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 25, 42], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.31520557403564453], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e51f83a0697a73211c79cf047dc8dd1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_564531cf15b0403b21d6702acac85d15
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 13, 21], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4658982455730438], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ff25a6bac4869539f2b3d4543908b386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_564531cf15b0403b21d6702acac85d15
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 7, 11], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4570605158805847], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6c5c0def5b7a7b7272da56fd3930640f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5444e154e808e8fd49fd1566aeebc9fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c5c0def5b7a7b7272da56fd3930640f
    def get_inputs(self):
        return [
            paddle.uniform([1, 16800, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 16800, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_77da853883fb6e7adc8a09266b7b7c1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c5c0def5b7a7b7272da56fd3930640f
    def get_inputs(self):
        return [
            paddle.uniform([1, 4200, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4200, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_77fbf7d642c96654ff76f7bc86861d7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c5c0def5b7a7b7272da56fd3930640f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1050, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1050, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7e47559b2a5c1396bf0c4f174296c166(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c5c0def5b7a7b7272da56fd3930640f
    def get_inputs(self):
        return [
            paddle.uniform([1, 273, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 273, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ceda610073ea4f5de741c3e31b396a14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c5c0def5b7a7b7272da56fd3930640f
    def get_inputs(self):
        return [
            paddle.uniform([1, 77, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 77, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8a2104239a5e321b6448140be3b02824(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 32, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5030e2d374a2282add66ce865275881a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a2104239a5e321b6448140be3b02824
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 28, 28], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_0431bb476024b2ced4ea4e95edddbf4b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 48, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bf48f6619e7e1a361d7cf8e7efbc1362(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0431bb476024b2ced4ea4e95edddbf4b
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 14, 14], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_771409a872cf2d4b6ed95a6beb1e2503(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 64, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5764efc4f2072ad0453bf868ac6e9a8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_771409a872cf2d4b6ed95a6beb1e2503
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 14, 14], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_5a5020b3b1b3280f186f2f53c631720b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 128, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_74766d2915a0a3a4969b49cff3757282(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a5020b3b1b3280f186f2f53c631720b
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 14, 14], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_19c6b50a613c857573cf0d7ac3c25d89(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 128, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ec2e44ceb90270cd647b36e9227412c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19c6b50a613c857573cf0d7ac3c25d89
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 7, 7], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_0d7a9581b43d62ce89fec8ec6f14f689(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 256, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_02c9a4409f4c4a1d110ed2c5dfa7a871(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d7a9581b43d62ce89fec8ec6f14f689
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 7, 7], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_06b3c6c1737c204eb86409faf8be5fc5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 196, 320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_78e107616040cf68a418c26820c75f96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06b3c6c1737c204eb86409faf8be5fc5
    def get_inputs(self):
        return [
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 196, 320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a5fb2c32c57db18a194db892dbf7df0e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 49, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d38ccdff1b60aa1ae2b1e4bac1d997b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5fb2c32c57db18a194db892dbf7df0e
    def get_inputs(self):
        return [
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 49, 512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_80ab364f5d9f994d2d4da0f908a32fca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 26, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 26, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d64e22beec703ec3207dbfb077ced613(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80ab364f5d9f994d2d4da0f908a32fca
    def get_inputs(self):
        return [
            paddle.uniform([1, 26, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 26, 512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_013a62a441952dc61230fc7213d89e05(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2, 1, 9, 112, 112], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 2, 16, 9, 112, 112], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9cf18d8c8041a22c5cf75d6efe751dc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_013a62a441952dc61230fc7213d89e05
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 1, 9, 112, 112], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 16, 9, 112, 112], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8808ee59a3fefd85be195f64ff5252ba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 1, 49, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4, 16, 49, 56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3f78caea6fada5159e6d6e5fd75a440f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8808ee59a3fefd85be195f64ff5252ba
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 1, 49, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 4, 16, 49, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_89061cf690c99483afdcbc56014d1920(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, 1, 49, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 8, 16, 49, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1c7b7d2514e3a0519225f7fd6a57522a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89061cf690c99483afdcbc56014d1920
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1, 49, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_609323b256d051b79c57abffc06ddbb0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 1, 49, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 16, 16, 49, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9fdee4aeada673b6258bb02c60ef98de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_609323b256d051b79c57abffc06ddbb0
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 1, 49, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_32efcbab1dedafbdb0f4c7dcc5166f72(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 1, 49, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 32, 16, 49, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0b5264c08d349f0ff226eceeeb130883(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32efcbab1dedafbdb0f4c7dcc5166f72
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 49, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5890a8f739978b9f14a329510c56be53(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 256, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d870fe2c4d7f02cdea85d168631a969e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5890a8f739978b9f14a329510c56be53
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 16, 16], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_d8683056bd0a66b8740146010186fed5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 512, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4476de9c278a9ff58bf7ce6bb80edc99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8683056bd0a66b8740146010186fed5
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 16, 16], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_45ed7893b2580aea9e2c98c46a69e12b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad96b4ff0519273f7db2993654355938
    def get_inputs(self):
        return [
            paddle.to_tensor([0.29906246066093445], dtype='float32').reshape([1]),
            paddle.uniform([1, 16, 320, 320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e52b38174a18cb03d203aeab77022a48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad96b4ff0519273f7db2993654355938
    def get_inputs(self):
        return [
            paddle.to_tensor([0.43402084708213806], dtype='float32').reshape([1]),
            paddle.uniform([1, 16, 320, 320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_60ea5f2063fb2da02bd447261a632f2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25a9b384a8df1f2393ddef8fd3a80bab
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2055632621049881], dtype='float32').reshape([1]),
            paddle.uniform([1, 32, 320, 320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6e81cd38fa9e9b634dc03345e1a6f2aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25a9b384a8df1f2393ddef8fd3a80bab
    def get_inputs(self):
        return [
            paddle.to_tensor([0.22573179006576538], dtype='float32').reshape([1]),
            paddle.uniform([1, 32, 320, 320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a9a18dc877165e468319c2d03287dc30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25a9b384a8df1f2393ddef8fd3a80bab
    def get_inputs(self):
        return [
            paddle.to_tensor([0.42056146264076233], dtype='float32').reshape([1]),
            paddle.uniform([1, 32, 160, 160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1fa5ca7e4b32527b45bd8e9e5e0a146f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6cbc81f9d908468332d470608147155
    def get_inputs(self):
        return [
            paddle.to_tensor([0.12126166373491287], dtype='float32').reshape([1]),
            paddle.uniform([1, 48, 160, 160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e5d48a2b8d8d5671b4a8e08e23e23dc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6cbc81f9d908468332d470608147155
    def get_inputs(self):
        return [
            paddle.to_tensor([0.09242968261241913], dtype='float32').reshape([1]),
            paddle.uniform([1, 48, 160, 160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8d57d64978a0c90d1fc70391891ed81d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6cbc81f9d908468332d470608147155
    def get_inputs(self):
        return [
            paddle.to_tensor([0.004355288110673428], dtype='float32').reshape([1]),
            paddle.uniform([1, 48, 160, 160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_821a15b8aa19a5233273470ce4678f61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6cbc81f9d908468332d470608147155
    def get_inputs(self):
        return [
            paddle.to_tensor([0.11184486001729965], dtype='float32').reshape([1]),
            paddle.uniform([1, 48, 160, 160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_363badbc890cd7c7778d4b460b7fa836(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6cbc81f9d908468332d470608147155
    def get_inputs(self):
        return [
            paddle.to_tensor([0.14176878333091736], dtype='float32').reshape([1]),
            paddle.uniform([1, 48, 160, 160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6f7e1b42b52f7825fce9c8581c6a4e5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6cbc81f9d908468332d470608147155
    def get_inputs(self):
        return [
            paddle.to_tensor([0.23379838466644287], dtype='float32').reshape([1]),
            paddle.uniform([1, 48, 160, 160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_83ae79cd6710c9d860efc459a22ee685(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6cbc81f9d908468332d470608147155
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3178041875362396], dtype='float32').reshape([1]),
            paddle.uniform([1, 48, 80, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8f33f25a5f165ebfd9df19c0bc6e10db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29a3dcdcc57b20a7a8622b91fac35f48
    def get_inputs(self):
        return [
            paddle.to_tensor([0.15914151072502136], dtype='float32').reshape([1]),
            paddle.uniform([1, 96, 80, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fe1ce57f2277b96b6d27348fe8aa692d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29a3dcdcc57b20a7a8622b91fac35f48
    def get_inputs(self):
        return [
            paddle.to_tensor([0.34217628836631775], dtype='float32').reshape([1]),
            paddle.uniform([1, 96, 80, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1ce7b8686d5efdb6da2c3c3f6e1c86d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29a3dcdcc57b20a7a8622b91fac35f48
    def get_inputs(self):
        return [
            paddle.to_tensor([0.21943950653076172], dtype='float32').reshape([1]),
            paddle.uniform([1, 96, 80, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e5e2425cc9b557e75dc55864d0d257bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29a3dcdcc57b20a7a8622b91fac35f48
    def get_inputs(self):
        return [
            paddle.to_tensor([0.046651214361190796], dtype='float32').reshape([1]),
            paddle.uniform([1, 96, 80, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7d14da9ee3b5611482488b6c8e91721f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29a3dcdcc57b20a7a8622b91fac35f48
    def get_inputs(self):
        return [
            paddle.to_tensor([0.23213328421115875], dtype='float32').reshape([1]),
            paddle.uniform([1, 96, 80, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_63d31ddec22aaecf744e2ad7c1d5e78f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29a3dcdcc57b20a7a8622b91fac35f48
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3153538405895233], dtype='float32').reshape([1]),
            paddle.uniform([1, 96, 80, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f3f98e08d2f78ba543a2517ffb0365c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29a3dcdcc57b20a7a8622b91fac35f48
    def get_inputs(self):
        return [
            paddle.to_tensor([0.09391975402832031], dtype='float32').reshape([1]),
            paddle.uniform([1, 96, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_05a861fc17d5e1ba399ee54cdaef7710(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a28e6e94ec21319929921665e8cfc9d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.07466598600149155], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f17283dc051a4cc59416384ea7ddc3cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a28e6e94ec21319929921665e8cfc9d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.358701229095459], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4d65550a4ae4b84453ade8be82d643cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a28e6e94ec21319929921665e8cfc9d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.09427876770496368], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_34e4931811be96dfa5121b8f414ff703(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a28e6e94ec21319929921665e8cfc9d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.49441027641296387], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9f255a1022fcd9797692694b83e38754(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a28e6e94ec21319929921665e8cfc9d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.46217530965805054], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7c6f4e6e68a23dd73f557e7910d090e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a28e6e94ec21319929921665e8cfc9d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0662069320678711], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_23ff62074dbdf897f41e68248854b21a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a28e6e94ec21319929921665e8cfc9d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1861724704504013], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_48ca50d4f31715a9a4e8f7a6bd5cb1cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a28e6e94ec21319929921665e8cfc9d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.06581340730190277], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2b7f362a631ac309a5e6be2ea18bfdd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a28e6e94ec21319929921665e8cfc9d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.06165476143360138], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e2f35338742c81b9a787e13e1ca52ec7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a28e6e94ec21319929921665e8cfc9d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.47823676466941833], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a71fe126d0657038b7f00c2cd6db6f56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a28e6e94ec21319929921665e8cfc9d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.46561551094055176], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cbfc2878baa9bf3e66b1dd23d7101f9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a28e6e94ec21319929921665e8cfc9d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1863097995519638], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_25f5a982a34343fb394375886c5a1ab5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a28e6e94ec21319929921665e8cfc9d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.056068096309900284], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dd2e2f9fe9f959a951ec1156a84d33db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a28e6e94ec21319929921665e8cfc9d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3534022569656372], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6fde553823e9851c32b132912858c99f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a28e6e94ec21319929921665e8cfc9d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.07152021676301956], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4e1a9c05170bb5672935c0359e813c55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a28e6e94ec21319929921665e8cfc9d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2648335099220276], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b6b63ea0200b2d9e155c5f872db63ffc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a28e6e94ec21319929921665e8cfc9d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.38199907541275024], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3b0bf64249f87779809c05e0080160e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a28e6e94ec21319929921665e8cfc9d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.07299169152975082], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f08fe43e84158c931fd8721bb5467f86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a28e6e94ec21319929921665e8cfc9d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3615790903568268], dtype='float32').reshape([1]),
            paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9bc80ab293204d0d438363d37422a238(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de55f9ce9eade27536df339da542b103
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_de8fd2cc0f6144145a15664e61bdc860(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22e2bd3a5a9602d84b8fae1fdff2989e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.21278151869773865], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_054da3b34e50fb8e844df304c763f46e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22e2bd3a5a9602d84b8fae1fdff2989e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.34000837802886963], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e1b5e3f53806ba2c6ac2f05f556e085e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22e2bd3a5a9602d84b8fae1fdff2989e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.31481823325157166], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f27f91688ff5c1ffc40042d55aa93287(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22e2bd3a5a9602d84b8fae1fdff2989e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.39017990231513977], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b2bacd056ff85252a4cf65f3c1841fc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7067e826a5fca59676079df4bf849961
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_01f61a1f2ebe3fbdbfa9094e1de092de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22e2bd3a5a9602d84b8fae1fdff2989e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2968728840351105], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a2b1ce13525552ff12d53de01fbde083(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22e2bd3a5a9602d84b8fae1fdff2989e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.046320926398038864], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a02e3716ac0b6ee8c47b8e44d14ddbbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22e2bd3a5a9602d84b8fae1fdff2989e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.20497341454029083], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5e079905ed4300a1084c789c96b771ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22e2bd3a5a9602d84b8fae1fdff2989e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.30073490738868713], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_24792bba3a9be625a78d08a1349a9783(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22e2bd3a5a9602d84b8fae1fdff2989e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.49956014752388], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c56bd393abcf30da903643e1a1e61446(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22e2bd3a5a9602d84b8fae1fdff2989e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.14100289344787598], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4c9a9ed7075cf29149b22a4264836bfd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22e2bd3a5a9602d84b8fae1fdff2989e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3750498294830322], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_05b9c265acef77487c5b3a147a9eda3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22e2bd3a5a9602d84b8fae1fdff2989e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.24916493892669678], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8611a7b62d6a59cbf29902c52c197c5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22e2bd3a5a9602d84b8fae1fdff2989e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10982408374547958], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7cfbcdc1ecafb59880df1ca6eaffc3e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22e2bd3a5a9602d84b8fae1fdff2989e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.34541022777557373], dtype='float32').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_99b4fc843983948d0e3ff011aa286d89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c8ef0676d9d343787fed8c6d03fea09
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 20, 20], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d5cb6432be38d629e833a7bf13e3b85f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c8ef0676d9d343787fed8c6d03fea09
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 40, 40], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_3449cb851373475c0b02b41e203a7877(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c8ef0676d9d343787fed8c6d03fea09
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 80, 80], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_066ba6ef01e3cfd7b8c8cf88f2f75ca8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c8ef0676d9d343787fed8c6d03fea09
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 160, 160], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_13fc4aa40be9339d6526330e33df40cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c8c92f07f7903913d2cbf14b8c2e5d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_be03902bc84ada3fa661021e36fb463d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c8c92f07f7903913d2cbf14b8c2e5d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_94145cc765b6dceef3ca8c8bb3df750d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c8c92f07f7903913d2cbf14b8c2e5d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 80, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9ee222b7cff2b461c49dd36ff8d92afb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7c8c92f07f7903913d2cbf14b8c2e5d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 160, 160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9d59da557f783a5ec3a3331d8579c777(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_986be6b479d0fbd9c1c1b95871b04b42
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 80, 80], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_d8587e96db215d370a0b5f5a2e57c23a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b34bb4ef69193a81fd98f1f0dd09f8c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 80, 80], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_f7fe0796d0a7be0dcd0c7d60be8888bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2bad6d29dc72b5bff76b4c6e1a5034e
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 40, 40], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_9f199caa2077c721e46cfe70156c3576(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af1528c59b4bcac06a62b37b374f0b10
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 40, 40], dtype='float16', min=0, max=0.5),
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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ec397a210b3e4252aed18cefa867f7fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af1528c59b4bcac06a62b37b374f0b10
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 20, 20], dtype='float16', min=0, max=0.5),
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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_63a4ccfe142638c7722680c3a996987c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c8b0b4c329c20902f2f56b6fa72d605
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 20, 20], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_807f7ea8556726a955d28cfdf334e1e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a25c7c6aeba2080e4c338ea05a69adf1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.38720703125], dtype='float16').reshape([1]),
            paddle.uniform([1, 16, 320, 320], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cc71de8ea7ebaa6eb10bda558865c9c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a25c7c6aeba2080e4c338ea05a69adf1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.09393310546875], dtype='float16').reshape([1]),
            paddle.uniform([1, 16, 320, 320], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_74172e67cd49aff3f292be65ea89e62b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d39d116a6d93012a5795633b24b57810
    def get_inputs(self):
        return [
            paddle.to_tensor([0.076416015625], dtype='float16').reshape([1]),
            paddle.uniform([1, 32, 320, 320], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_608ed4ee3cbe11bfe2289c26cba0cc91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d39d116a6d93012a5795633b24b57810
    def get_inputs(self):
        return [
            paddle.to_tensor([0.06378173828125], dtype='float16').reshape([1]),
            paddle.uniform([1, 32, 320, 320], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d162746a536126b346bfa05f92f6bfa8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d39d116a6d93012a5795633b24b57810
    def get_inputs(self):
        return [
            paddle.to_tensor([0.44140625], dtype='float16').reshape([1]),
            paddle.uniform([1, 32, 160, 160], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7b3a975736fb52711e6b0355123684a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe3fca136e37242dc43c1fbccc2fde49
    def get_inputs(self):
        return [
            paddle.to_tensor([0.48828125], dtype='float16').reshape([1]),
            paddle.uniform([1, 48, 160, 160], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7726c696fd6b7e2033bd562cb1ff9758(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe3fca136e37242dc43c1fbccc2fde49
    def get_inputs(self):
        return [
            paddle.to_tensor([0.375], dtype='float16').reshape([1]),
            paddle.uniform([1, 48, 160, 160], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_80844f900108d563ef1bc9f453ab8496(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe3fca136e37242dc43c1fbccc2fde49
    def get_inputs(self):
        return [
            paddle.to_tensor([0.388916015625], dtype='float16').reshape([1]),
            paddle.uniform([1, 48, 160, 160], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5bc32773f5f5554734381ab9bd87ba67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe3fca136e37242dc43c1fbccc2fde49
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1700439453125], dtype='float16').reshape([1]),
            paddle.uniform([1, 48, 160, 160], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bebb5b5a7a435366e9c306521af7ba81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe3fca136e37242dc43c1fbccc2fde49
    def get_inputs(self):
        return [
            paddle.to_tensor([0.393310546875], dtype='float16').reshape([1]),
            paddle.uniform([1, 48, 160, 160], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ead4ea32a799b96698ae8d0cdce99a7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe3fca136e37242dc43c1fbccc2fde49
    def get_inputs(self):
        return [
            paddle.to_tensor([0.42626953125], dtype='float16').reshape([1]),
            paddle.uniform([1, 48, 160, 160], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a2ccd281834a1dc24cf5cc8071363b50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe3fca136e37242dc43c1fbccc2fde49
    def get_inputs(self):
        return [
            paddle.to_tensor([0.30810546875], dtype='float16').reshape([1]),
            paddle.uniform([1, 48, 80, 80], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_078eb525a3e0ad242e4f94cad797e7f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c45923d4d5fc586b60fc1bb097fed418
    def get_inputs(self):
        return [
            paddle.to_tensor([0.33984375], dtype='float16').reshape([1]),
            paddle.uniform([1, 96, 80, 80], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6eeb7e440a4a59f07e724f98345102c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c45923d4d5fc586b60fc1bb097fed418
    def get_inputs(self):
        return [
            paddle.to_tensor([0.30517578125], dtype='float16').reshape([1]),
            paddle.uniform([1, 96, 80, 80], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5c838235a9b58b036da6691cfdbdf9d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c45923d4d5fc586b60fc1bb097fed418
    def get_inputs(self):
        return [
            paddle.to_tensor([0.043670654296875], dtype='float16').reshape([1]),
            paddle.uniform([1, 96, 80, 80], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4cdceed8e25680653a63f097f37e6bd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c45923d4d5fc586b60fc1bb097fed418
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3349609375], dtype='float16').reshape([1]),
            paddle.uniform([1, 96, 80, 80], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f9bac967de3204ad996ce54891707e04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c45923d4d5fc586b60fc1bb097fed418
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2587890625], dtype='float16').reshape([1]),
            paddle.uniform([1, 96, 80, 80], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fb66098ac8fe01b676ee78e39670703e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c45923d4d5fc586b60fc1bb097fed418
    def get_inputs(self):
        return [
            paddle.to_tensor([0.14208984375], dtype='float16').reshape([1]),
            paddle.uniform([1, 96, 80, 80], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fac8b754c21e9e3105dd0806d3e8d185(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c45923d4d5fc586b60fc1bb097fed418
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0240478515625], dtype='float16').reshape([1]),
            paddle.uniform([1, 96, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fb5f572f6ef9fc7911e3c782d6072086(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_689176cc2300dfd17d02a2f49a9cb30c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.146484375], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f38676d9441f41bde43b71bb9066e194(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_689176cc2300dfd17d02a2f49a9cb30c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1553955078125], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_75885abdc5cbe846c6a39a26619a0b36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_689176cc2300dfd17d02a2f49a9cb30c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.056915283203125], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d4441027543a98a6cb1142f2ab9fa506(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_689176cc2300dfd17d02a2f49a9cb30c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.07470703125], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_688ba1836df0aa668e44e263332c7805(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_689176cc2300dfd17d02a2f49a9cb30c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.429443359375], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b89ec00af3f9944e98895c56e35f4f1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_689176cc2300dfd17d02a2f49a9cb30c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.036529541015625], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d666966414783ef23fc21435a150f28f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_689176cc2300dfd17d02a2f49a9cb30c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.09844970703125], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ffe7490e1bd904368e3c50d3c823f533(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_689176cc2300dfd17d02a2f49a9cb30c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.348876953125], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7a682c112928a4fa0f0c1f538d78883d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_689176cc2300dfd17d02a2f49a9cb30c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.032470703125], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ebe3e617a0f172d55f3fe3c538a9c16a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_689176cc2300dfd17d02a2f49a9cb30c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2406005859375], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_22a985bd69cc4fb6f618fc5febefd3c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_689176cc2300dfd17d02a2f49a9cb30c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.32177734375], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8f902a8b10302b245e14e2d09427be17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_689176cc2300dfd17d02a2f49a9cb30c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.34619140625], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_93691860b8943399998c2092872eaa14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_689176cc2300dfd17d02a2f49a9cb30c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.434326171875], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b4db8d87317505484bfcc2696a6ccc90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_689176cc2300dfd17d02a2f49a9cb30c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.038543701171875], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8124fd2f6e8bc460109d761ffc47a76b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_689176cc2300dfd17d02a2f49a9cb30c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.39453125], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9011584cf8464b623a100a391215acd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_689176cc2300dfd17d02a2f49a9cb30c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0390625], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a423098317b3949a189faacb04daefea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_689176cc2300dfd17d02a2f49a9cb30c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4619140625], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_aa6267cc69a5791ae874dcdbf19e78c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_689176cc2300dfd17d02a2f49a9cb30c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.049224853515625], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7feb7b057686f974482eb94be123a79b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_689176cc2300dfd17d02a2f49a9cb30c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.387939453125], dtype='float16').reshape([1]),
            paddle.uniform([1, 192, 20, 20], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ded03ac989ab731d5927dea5bb49432f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63b159b310cc8c7c69a8fd046ebd1b5e
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 20, 20], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_c41881a011c94cb2ec3074515f22a0f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0dd8f88cb736ffc47e4f6d2556dc562
    def get_inputs(self):
        return [
            paddle.to_tensor([0.06439208984375], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1dd603462e4c9b0aef4c92a86f0fe4e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0dd8f88cb736ffc47e4f6d2556dc562
    def get_inputs(self):
        return [
            paddle.to_tensor([0.492431640625], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_175b726a9e568b543bfcdef514a155df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0dd8f88cb736ffc47e4f6d2556dc562
    def get_inputs(self):
        return [
            paddle.to_tensor([0.29638671875], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9dcb0866fb9ecd4d71f1b6946d0c8f50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0dd8f88cb736ffc47e4f6d2556dc562
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10614013671875], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8d4c9da70bcb1c63402678168da65622(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2935bfe3850f9f8ad92d3dc68a8ad941
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 20, 20], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_a3f45b2385ec7b8c235f47836dfbcd38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0dd8f88cb736ffc47e4f6d2556dc562
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1583251953125], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_91873f798b300e78bce94165edc377ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0dd8f88cb736ffc47e4f6d2556dc562
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1260986328125], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3dd937cdf962b844519743c8006986f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0dd8f88cb736ffc47e4f6d2556dc562
    def get_inputs(self):
        return [
            paddle.to_tensor([0.259765625], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_26700c3f1c98c369509d36c9c8d44f1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0dd8f88cb736ffc47e4f6d2556dc562
    def get_inputs(self):
        return [
            paddle.to_tensor([0.158447265625], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_11044d057ecbf2b8358707becd2e626a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0dd8f88cb736ffc47e4f6d2556dc562
    def get_inputs(self):
        return [
            paddle.to_tensor([0.12200927734375], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bfe18e249c7cc22f4cd82f36679abef3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0dd8f88cb736ffc47e4f6d2556dc562
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1407470703125], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0acd57796bce33ec959963631b3f2cc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0dd8f88cb736ffc47e4f6d2556dc562
    def get_inputs(self):
        return [
            paddle.to_tensor([0.06878662109375], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7f651a1331357a88ac476b8349cee870(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0dd8f88cb736ffc47e4f6d2556dc562
    def get_inputs(self):
        return [
            paddle.to_tensor([0.056671142578125], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_68c5929dee4015b4d1184789f9cbd4bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0dd8f88cb736ffc47e4f6d2556dc562
    def get_inputs(self):
        return [
            paddle.to_tensor([0.01555633544921875], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2827a65d7de2269a40849082798e554a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0dd8f88cb736ffc47e4f6d2556dc562
    def get_inputs(self):
        return [
            paddle.to_tensor([0.429931640625], dtype='float16').reshape([1]),
            paddle.uniform([1, 384, 20, 20], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ef4b4190e3955caa6b2d560c8ecc88ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32276cd30b45b13643f12def8f1074f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 20, 20], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_b0610dcd7e748763cb184fbc7207ed25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32276cd30b45b13643f12def8f1074f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 40, 40], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_25a5d659d8b22efb690aa83097b59eb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32276cd30b45b13643f12def8f1074f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 80, 80], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_3cd84d834042b21fba1944f2307c468f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32276cd30b45b13643f12def8f1074f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 160, 160], dtype='float16', min=0, max=0.5),
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
class TestPrimitiveOp_1702caf01600f82f81631805461b807c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c54c838f0e6717b6c3138519561c831
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 20, 20], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float16').reshape([1, 24, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5229800893f66d4cf2b01e14bfd5e24c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c54c838f0e6717b6c3138519561c831
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 40, 40], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float16').reshape([1, 24, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3b0f78165d5bed47ff2103966e398856(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c54c838f0e6717b6c3138519561c831
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 80, 80], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float16').reshape([1, 24, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b03162b62039a58299ac1f5e104cd9c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c54c838f0e6717b6c3138519561c831
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 160, 160], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float16').reshape([1, 24, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4a4d946fe685cca9476a486a80f49e85(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[32, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 32, 56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_47c81d97d10687ab7802c2811c473dce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a4d946fe685cca9476a486a80f49e85
    def get_inputs(self):
        return [
            paddle.uniform([32, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 32, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a2635553381f6261f26afd13ca91535e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[64, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 64, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ab81738498a96e3aa5c4cfebcbfef658(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2635553381f6261f26afd13ca91535e
    def get_inputs(self):
        return [
            paddle.uniform([64, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 64, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_35ab96ed47b653129dffe1d312c2b131(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[160, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 160, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_44c05d5dafa8731505dd3151feb8867f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35ab96ed47b653129dffe1d312c2b131
    def get_inputs(self):
        return [
            paddle.uniform([160, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 160, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_89039391ffc64906a5a17271e6fad899(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[256, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 256, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ded9cfe41596a61fafce0cf915005180(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89039391ffc64906a5a17271e6fad899
    def get_inputs(self):
        return [
            paddle.uniform([256, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_255eedc6001202414d4278aaa5602282(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2fedc85369ffb9ecc024553605154e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 80, 80], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_aec09aa13d44029e87d6620aa77642a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_959b775eab74bbd70138b39517e94f07
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 80, 80], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_838a47d17d01555cf8a7488fbe9e04c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9235e831308f4e9c7ba9e3b8d4a4407c
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 40, 40], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_cb88fac39bf6c46d41c12764bb3ca48b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd264045cb0e22f7c3d28013ff3477dd
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 40, 40], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_852af1ddefca36889cff2beedb637b44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd264045cb0e22f7c3d28013ff3477dd
    def get_inputs(self):
        return [
            paddle.uniform([1, 336, 20, 20], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_3337f66801f1f929b7c55b5d7bfef48e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_701e5b72f8d7a1806675c65bd1524ad3
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 20, 20], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_d7d432edc71699916b47004a23ea0f7d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 384, 20, 20], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 384, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3f90d525096f7a0e8c145c168bcc8abd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7d432edc71699916b47004a23ea0f7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 20, 20], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_844abf5dab86ada9560b4cf5209fed80(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, 40, 40], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 192, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0068cfa6099ca6ac45370e7c9eb6aa27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_844abf5dab86ada9560b4cf5209fed80
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 40, 40], dtype='float16', min=0, max=0.5),
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

class PrimitiveOp_4a22ad653592862b2c6d4758407d9a73(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.multiply(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 80, 80], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 96, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_588f581e99abef57c849c521f4515b10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a22ad653592862b2c6d4758407d9a73
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 80, 80], dtype='float16', min=0, max=0.5),
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


if __name__ == '__main__':
    unittest.main()