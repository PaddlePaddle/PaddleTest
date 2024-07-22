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
class PrimitiveOp_e0357accc5b8e0076cf96e4b7ab69122(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([[0, 1]], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.yolo_box(input_0, input_1, [116, 90, 156, 198, 373, 326], 80, float('0.005'), 32, True, float('1'), False, float('0.5')), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d8adb0a413d5b20fcfcb8680b8a5061d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0357accc5b8e0076cf96e4b7ab69122
    def get_inputs(self):
        return [
            paddle.uniform([1, 255, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0, 1]], dtype='int32').reshape([1, 2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5827e006b57f290f477cef404f508485(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([[0, 1]], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.yolo_box(input_0, input_1, [30, 61, 62, 45, 59, 119], 80, float('0.005'), 16, True, float('1'), False, float('0.5')), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5fe6f45d4717f03109ee00ff06d24e15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5827e006b57f290f477cef404f508485
    def get_inputs(self):
        return [
            paddle.uniform([1, 255, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0, 1]], dtype='int32').reshape([1, 2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_dea7ef35dc40c108286c67cbdc8d015e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([[0, 1]], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.yolo_box(input_0, input_1, [10, 13, 16, 30, 33, 23], 80, float('0.005'), 8, True, float('1'), False, float('0.5')), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_85a04bd9c877887b4525f574573bfb57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dea7ef35dc40c108286c67cbdc8d015e
    def get_inputs(self):
        return [
            paddle.uniform([1, 255, 76, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0, 1]], dtype='int32').reshape([1, 2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_3ab37b78ab189a643c656d6f8445dcb6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([[1, 1]], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.yolo_box(input_0, input_1, [116, 90, 156, 198, 373, 326], 80, float('0.005'), 32, True, float('1'), False, float('0.5')), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fc03e851b57447e7f24d94ae6ab82262(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ab37b78ab189a643c656d6f8445dcb6
    def get_inputs(self):
        return [
            paddle.uniform([1, 255, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1, 1]], dtype='int32').reshape([1, 2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_fe5ebe362fe831d3b8aaa36c027d9361(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([[1, 1]], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.yolo_box(input_0, input_1, [30, 61, 62, 45, 59, 119], 80, float('0.005'), 16, True, float('1'), False, float('0.5')), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d73ee8065a3374ef5c3cf5a6830cfc76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe5ebe362fe831d3b8aaa36c027d9361
    def get_inputs(self):
        return [
            paddle.uniform([1, 255, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1, 1]], dtype='int32').reshape([1, 2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ac6bb4aaf0d22b62425456ab0667bad0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([[1, 1]], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.yolo_box(input_0, input_1, [10, 13, 16, 30, 33, 23], 80, float('0.005'), 8, True, float('1'), False, float('0.5')), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_740a7d9d7d60ee93837138d6270889be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac6bb4aaf0d22b62425456ab0667bad0
    def get_inputs(self):
        return [
            paddle.uniform([1, 255, 76, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1, 1]], dtype='int32').reshape([1, 2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9e81d35f14ac987c85c863d6d16b3957(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([[0, 1]], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.yolo_box(input_0, input_1, [116, 90, 156, 198, 373, 326], 80, float('0.005'), 32, True, float('1'), False, float('0.5')), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 255, 19, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_66abd3a643e491846983da70ec0be1e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e81d35f14ac987c85c863d6d16b3957
    def get_inputs(self):
        return [
            paddle.uniform([1, 255, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0, 1]], dtype='int32').reshape([1, 2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_164b9b69d073ca773bec4e005348b08e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([[0, 1]], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.yolo_box(input_0, input_1, [30, 61, 62, 45, 59, 119], 80, float('0.005'), 16, True, float('1'), False, float('0.5')), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 255, 38, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dc07c383cd6c2ebec760a96a15ee8203(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_164b9b69d073ca773bec4e005348b08e
    def get_inputs(self):
        return [
            paddle.uniform([1, 255, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0, 1]], dtype='int32').reshape([1, 2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_df893a3e2ed401ef8e4cec354f984759(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([[0, 1]], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.yolo_box(input_0, input_1, [10, 13, 16, 30, 33, 23], 80, float('0.005'), 8, True, float('1'), False, float('0.5')), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 255, 76, 76], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e1b7f562d70bdd9ba941caa4b1e5258d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df893a3e2ed401ef8e4cec354f984759
    def get_inputs(self):
        return [
            paddle.uniform([1, 255, 76, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0, 1]], dtype='int32').reshape([1, 2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e523f4c81a170add1490f98b457f889c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([[1, 1]], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.yolo_box(input_0, input_1, [116, 90, 156, 198, 373, 326], 80, float('0.005'), 32, True, float('1'), False, float('0.5')), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 255, 19, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_214bffb127625e13f25a4d6403d35732(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e523f4c81a170add1490f98b457f889c
    def get_inputs(self):
        return [
            paddle.uniform([1, 255, 19, 19], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1, 1]], dtype='int32').reshape([1, 2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_02ed9a982c400b71a905bfac98672cfd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([[1, 1]], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.yolo_box(input_0, input_1, [30, 61, 62, 45, 59, 119], 80, float('0.005'), 16, True, float('1'), False, float('0.5')), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 255, 38, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f9406b84c535471fdae254cd1a5ed743(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02ed9a982c400b71a905bfac98672cfd
    def get_inputs(self):
        return [
            paddle.uniform([1, 255, 38, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1, 1]], dtype='int32').reshape([1, 2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_63f5b5c0d68ba9a24a9d65f54f8e16c3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([[1, 1]], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.yolo_box(input_0, input_1, [10, 13, 16, 30, 33, 23], 80, float('0.005'), 8, True, float('1'), False, float('0.5')), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 255, 76, 76], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c5222f32d63fe344f51b16db94b0d50d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63f5b5c0d68ba9a24a9d65f54f8e16c3
    def get_inputs(self):
        return [
            paddle.uniform([1, 255, 76, 76], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1, 1]], dtype='int32').reshape([1, 2]),
        ]


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