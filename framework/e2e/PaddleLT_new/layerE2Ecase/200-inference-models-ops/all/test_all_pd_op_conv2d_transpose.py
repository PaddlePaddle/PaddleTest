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
class PrimitiveOp_0cdcf5323d0602458ac750ab381dc96a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        arg_2 = paddle._C_ops.full_int_array([], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return paddle._C_ops.conv2d_transpose(input_0, input_1, [2, 2], [1, 1], [1, 1], input_2, 'EXPLICIT', 1, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_780c65f48b317116827178965c1d44f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cdcf5323d0602458ac750ab381dc96a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 128, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fe43977f8e3732bed3decc1b00a6d12a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0cdcf5323d0602458ac750ab381dc96a
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 64, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_18d9d467002cd25d136d7b371e3bf071(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        arg_2 = paddle._C_ops.full_int_array([], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return paddle._C_ops.conv2d_transpose(input_0, input_1, [2, 2], [0, 0], [], input_2, 'EXPLICIT', 1, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b86540f01158ddad34c8d87187f24515(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_18d9d467002cd25d136d7b371e3bf071
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 160, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 24, 2, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fc6ee0596102248db74c456c1ee4c5f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_18d9d467002cd25d136d7b371e3bf071
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 320, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 1, 2, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1c0e7b8cade7b7eff5520e31d705eb64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_18d9d467002cd25d136d7b371e3bf071
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 240, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 24, 2, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_726ea1948c1cd3bee1ac6785de2b5c99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_18d9d467002cd25d136d7b371e3bf071
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 480, 480], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 1, 2, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_671df7f153e19813a9e68ddf7345443b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        arg_2 = paddle._C_ops.full_int_array([256, 512], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return paddle._C_ops.conv2d_transpose(input_0, input_1, [2, 2], [0, 0], [], input_2, 'EXPLICIT', 1, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cfbb37a7aedeff09cbf3234ee890babb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_671df7f153e19813a9e68ddf7345443b
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 32, 2, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([256, 512], dtype='int64').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_48a4d97dd802d3cf482804d77100e6de(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        arg_2 = paddle._C_ops.full_int_array([512, 1024], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return paddle._C_ops.conv2d_transpose(input_0, input_1, [2, 2], [0, 0], [], input_2, 'EXPLICIT', 1, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_62657f53059661174d4ff2db8fbdf074(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48a4d97dd802d3cf482804d77100e6de
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 16, 2, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([512, 1024], dtype='int64').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_dded65e11ca751f6e0d11fe92baf8ad5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        arg_2 = paddle._C_ops.full_int_array([1024, 2048], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return paddle._C_ops.conv2d_transpose(input_0, input_1, [2, 2], [1, 1], [], input_2, 'EXPLICIT', 1, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_75cc1aab41eecb2516395bfda31622b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dded65e11ca751f6e0d11fe92baf8ad5
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 512, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 19, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1024, 2048], dtype='int64').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5c40d50180bd0128fe6d07122c97d52d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        arg_2 = paddle._C_ops.full_int_array([], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return paddle._C_ops.conv2d_transpose(input_0, input_1, [2, 2], [1, 1], [], input_2, 'EXPLICIT', 1, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ef2b59ec8a9583c78a1ef5367814b28c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c40d50180bd0128fe6d07122c97d52d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_194b56bf4d8348be718bcc027e613f10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c40d50180bd0128fe6d07122c97d52d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 2, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 512, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8e60c4da6da11b3deb184356c2918c78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c40d50180bd0128fe6d07122c97d52d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 512, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_43cc655e81b2fcee2454edf651dfe652(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c40d50180bd0128fe6d07122c97d52d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 512, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cd434ff2c8a14b9ca1c053c6159058ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c40d50180bd0128fe6d07122c97d52d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 256, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cf0da099221a90adfab9ba8ad9e98139(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c40d50180bd0128fe6d07122c97d52d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 128, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d5109d8578c8cba6fe137a21243d640b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c40d50180bd0128fe6d07122c97d52d
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 64, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f7c6648266429b98937d3f70ac1d37d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c40d50180bd0128fe6d07122c97d52d
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 3, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_289dfdc63ee1e6b21379240c6baf04d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c40d50180bd0128fe6d07122c97d52d
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d339f404f1ac125fe6cc98020d5c2301(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c40d50180bd0128fe6d07122c97d52d
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 192, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6990672bec44c892ebe240ef2d9df1a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c40d50180bd0128fe6d07122c97d52d
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ba4bff88eaf75b48c1155e2be4be623e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c40d50180bd0128fe6d07122c97d52d
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 80, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 128, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8bfd8ac99bb61857d1c5987a58a119ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c40d50180bd0128fe6d07122c97d52d
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5adbc53eb95a231b7f68dc795396ac4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c40d50180bd0128fe6d07122c97d52d
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 192, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4c6f838d854efa93c4278a36af9f0de5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c40d50180bd0128fe6d07122c97d52d
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dc958d8611b46ab62581dcb747d039a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c40d50180bd0128fe6d07122c97d52d
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 120, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 128, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_395c0f26253831ed56ba6257af9996c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_18d9d467002cd25d136d7b371e3bf071
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 240, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 2, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a32a68275bb9be73176766fd0450e219(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_18d9d467002cd25d136d7b371e3bf071
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 480, 480], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 1, 2, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8c226f54da208eff4e1fb3a6f746973a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_18d9d467002cd25d136d7b371e3bf071
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 160, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 2, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7997990cf446d3fc7491a29c7eeb558e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_18d9d467002cd25d136d7b371e3bf071
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 320, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 1, 2, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0c54e462e3b469f126487ef904270ca4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        arg_2 = paddle._C_ops.full_int_array([], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return paddle._C_ops.conv2d_transpose(input_0, input_1, [2, 2], [1, 1], [1, 1], input_2, 'EXPLICIT', 1, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 128, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[0], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a636bc3d056658ad89d7f83e542f05ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c54e462e3b469f126487ef904270ca4
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 128, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b87b1a301a26228535d4eee38484437a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        arg_2 = paddle._C_ops.full_int_array([], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return paddle._C_ops.conv2d_transpose(input_0, input_1, [2, 2], [1, 1], [1, 1], input_2, 'EXPLICIT', 1, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 64, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[0], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f672eaabef15a16b48ecd86e4cfb4311(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b87b1a301a26228535d4eee38484437a
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 64, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_061cb3fbf22aae8b310bd6bfe3ec744d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        arg_2 = paddle._C_ops.full_int_array([], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return paddle._C_ops.conv2d_transpose(input_0, input_1, [2, 2], [0, 0], [], input_2, 'EXPLICIT', 1, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 24, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[24, 24, 2, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[0], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_545ca95f0222066f95b77ea38fb89a3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_061cb3fbf22aae8b310bd6bfe3ec744d
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 160, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 24, 2, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_595922099c31cd5583633af43acfe718(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        arg_2 = paddle._C_ops.full_int_array([], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return paddle._C_ops.conv2d_transpose(input_0, input_1, [2, 2], [0, 0], [], input_2, 'EXPLICIT', 1, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 24, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[24, 1, 2, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[0], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f57ace170da29c1b514f5bc95be85efd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_595922099c31cd5583633af43acfe718
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 320, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 1, 2, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a4abb320d10f5d3e984c5bd472a49686(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_061cb3fbf22aae8b310bd6bfe3ec744d
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 240, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 24, 2, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_74cd9f05c75c4e1176b684e9bcf2f534(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_595922099c31cd5583633af43acfe718
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 480, 480], dtype='float32', min=0, max=0.5),
            paddle.uniform([24, 1, 2, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5b2976d19c1c2b45fd4a1ac445a219a3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        arg_2 = paddle._C_ops.full_int_array([256, 512], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return paddle._C_ops.conv2d_transpose(input_0, input_1, [2, 2], [0, 0], [], input_2, 'EXPLICIT', 1, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 32, 128, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[32, 32, 2, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_db5406d611737a0c587b65e3df1b9c0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b2976d19c1c2b45fd4a1ac445a219a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([32, 32, 2, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([256, 512], dtype='int64').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_938153cd43819c22dde4b48c79d1575c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        arg_2 = paddle._C_ops.full_int_array([512, 1024], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return paddle._C_ops.conv2d_transpose(input_0, input_1, [2, 2], [0, 0], [], input_2, 'EXPLICIT', 1, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 16, 256, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[16, 16, 2, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7563a5417d150b43970c2067330305f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_938153cd43819c22dde4b48c79d1575c
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 16, 2, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([512, 1024], dtype='int64').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_37232c87ebc09023fdab08beae4f51d7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        arg_2 = paddle._C_ops.full_int_array([1024, 2048], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return paddle._C_ops.conv2d_transpose(input_0, input_1, [2, 2], [1, 1], [], input_2, 'EXPLICIT', 1, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 16, 512, 1024], dtype='float32'),
            paddle.static.InputSpec(shape=[16, 19, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7f8a1a3791fb6d771b6ae2c299913494(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37232c87ebc09023fdab08beae4f51d7
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 512, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([16, 19, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1024, 2048], dtype='int64').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_62a32e2043ddad5a9c2a56a164a1a61f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        arg_2 = paddle._C_ops.full_int_array([], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return paddle._C_ops.conv2d_transpose(input_0, input_1, [2, 2], [1, 1], [], input_2, 'EXPLICIT', 1, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 512, 4, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[0], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_25cb92c980957b455718d6e3ab37e915(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62a32e2043ddad5a9c2a56a164a1a61f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a2366de95f58be2d2d4f5d324a5b50ac(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        arg_2 = paddle._C_ops.full_int_array([], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return paddle._C_ops.conv2d_transpose(input_0, input_1, [2, 2], [1, 1], [], input_2, 'EXPLICIT', 1, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1024, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1024, 512, 4, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[0], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1aaabbdbe6867aaedf3cec15ede28542(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2366de95f58be2d2d4f5d324a5b50ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 2, 2], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 512, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9b74b7a838483ccd6c650ab8657b7b23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2366de95f58be2d2d4f5d324a5b50ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 512, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_613eb2fb09826e06b1e1f859a5be7fa6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2366de95f58be2d2d4f5d324a5b50ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 512, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_94d90b07dd1c7f11724aaa88a25df389(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        arg_2 = paddle._C_ops.full_int_array([], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return paddle._C_ops.conv2d_transpose(input_0, input_1, [2, 2], [1, 1], [], input_2, 'EXPLICIT', 1, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1024, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1024, 256, 4, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[0], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_44a30db6f6f8a7703edf725ddf0fb77a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94d90b07dd1c7f11724aaa88a25df389
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024, 256, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f92815b0c573261a81ca04432385660a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        arg_2 = paddle._C_ops.full_int_array([], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return paddle._C_ops.conv2d_transpose(input_0, input_1, [2, 2], [1, 1], [], input_2, 'EXPLICIT', 1, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 128, 4, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[0], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6002242c7da62bb9752796c8bb5c0985(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f92815b0c573261a81ca04432385660a
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 128, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a2e2f0f9863fc583dfa1a19f018d98af(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        arg_2 = paddle._C_ops.full_int_array([], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return paddle._C_ops.conv2d_transpose(input_0, input_1, [2, 2], [1, 1], [], input_2, 'EXPLICIT', 1, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 64, 4, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[0], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_988464433faa3a85db7a2fe6530596ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2e2f0f9863fc583dfa1a19f018d98af
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 64, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_53740f9b21f5da439d3ee04ab52a8830(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        arg_2 = paddle._C_ops.full_int_array([], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return paddle._C_ops.conv2d_transpose(input_0, input_1, [2, 2], [1, 1], [], input_2, 'EXPLICIT', 1, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 3, 4, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[0], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6c284514e75e97eaefd60162b4bd31e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53740f9b21f5da439d3ee04ab52a8830
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 3, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_bdf15fba0e1f8d8ebdc3112c3648315a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        arg_2 = paddle._C_ops.full_int_array([], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return paddle._C_ops.conv2d_transpose(input_0, input_1, [2, 2], [1, 1], [], input_2, 'EXPLICIT', 1, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 256, 4, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[0], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a4f60a28696489e33c5eb32f57efc2b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bdf15fba0e1f8d8ebdc3112c3648315a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_783b7c1d6eabf35b0291e85a2a3a2d84(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        arg_2 = paddle._C_ops.full_int_array([], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return paddle._C_ops.conv2d_transpose(input_0, input_1, [2, 2], [1, 1], [], input_2, 'EXPLICIT', 1, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 192, 4, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[0], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8b7addfb14c6ca22183502fbe69f0f77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_783b7c1d6eabf35b0291e85a2a3a2d84
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 192, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_88f4da7b87e6f94ee438f7d574a19e07(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        arg_2 = paddle._C_ops.full_int_array([], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return paddle._C_ops.conv2d_transpose(input_0, input_1, [2, 2], [1, 1], [], input_2, 'EXPLICIT', 1, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 192, 4, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[0], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4e66bc9792a11e200a59755602eed42b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88f4da7b87e6f94ee438f7d574a19e07
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_50e4152260a5eb36ea2ee612ff2a2193(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        arg_2 = paddle._C_ops.full_int_array([], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return paddle._C_ops.conv2d_transpose(input_0, input_1, [2, 2], [1, 1], [], input_2, 'EXPLICIT', 1, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[192, 128, 4, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[0], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0c7a01f57809173c7e946dc62569f9de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50e4152260a5eb36ea2ee612ff2a2193
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 80, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 128, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c7d3302642f8ea0128bc50e2235fe3f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bdf15fba0e1f8d8ebdc3112c3648315a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 15, 15], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_aa0a63d2793a641fe710bc01c8eb0dc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_783b7c1d6eabf35b0291e85a2a3a2d84
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 30, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 192, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_734afaeb6a7feac324a9785948cc849b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88f4da7b87e6f94ee438f7d574a19e07
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 60, 60], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 192, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c3c34e66c84670f3ecdcdd32b2718a37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50e4152260a5eb36ea2ee612ff2a2193
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 120, 120], dtype='float32', min=0, max=0.5),
            paddle.uniform([192, 128, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5ce39b403a74d15c54f6b8ff693cdbc3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        arg_2 = paddle._C_ops.full_int_array([], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return paddle._C_ops.conv2d_transpose(input_0, input_1, [2, 2], [0, 0], [], input_2, 'EXPLICIT', 1, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 64, 2, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[0], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_422c71fa05cc300c1bfe17ec58f62352(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ce39b403a74d15c54f6b8ff693cdbc3
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 240, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 2, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_cd396c7dc94fbb685f45349ef1668700(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        arg_2 = paddle._C_ops.full_int_array([], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return paddle._C_ops.conv2d_transpose(input_0, input_1, [2, 2], [0, 0], [], input_2, 'EXPLICIT', 1, [1, 1], 'NCHW')

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[64, 1, 2, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[0], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5058da48350e627c657c4e2137941709(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd396c7dc94fbb685f45349ef1668700
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 480, 480], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 1, 2, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_682bdf3f345cb5ea4106637de2498a71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ce39b403a74d15c54f6b8ff693cdbc3
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 160, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 64, 2, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_85412b510c92920e9227cf94c9f32207(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd396c7dc94fbb685f45349ef1668700
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 320, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([64, 1, 2, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([], dtype='int64').reshape([0]),
        ]


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