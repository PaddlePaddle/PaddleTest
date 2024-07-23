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
class PrimitiveOp_935be8ec86ea0846f90144c0da32c940(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2_0, arg_3_0, arg_4):
        arg_2_0 = paddle._C_ops.full_int_array(0, paddle.int64, paddle.core.CPUPlace())
        arg_3_0 = paddle._C_ops.full_int_array(1, paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = [arg_2_0]
        input_3 = [arg_3_0]
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor(input_0, input_1, input_2, input_3, input_4, [1], [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_279af411fd183c8d4752f7142b5b83e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_935be8ec86ea0846f90144c0da32c940
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 256], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor(0, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c6d2aa12b1e7c95db4ba5ae21f76f2b4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2_0, arg_3_0, arg_4):
        arg_2_0 = paddle._C_ops.full_int_array(0, paddle.int64, paddle.core.CPUPlace())
        arg_3_0 = paddle._C_ops.full_int_array(1, paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = [arg_2_0]
        input_3 = [arg_3_0]
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor(input_0, input_1, input_2, input_3, input_4, [1], [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8f70860832bdfa04cedce285e348979d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6d2aa12b1e7c95db4ba5ae21f76f2b4
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(0, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_543007accbe62dc7023e6af5a9fd6c90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6d2aa12b1e7c95db4ba5ae21f76f2b4
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 30], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[16.966392517089844, 16.338186264038086, 15.731739044189453, 15.963295936584473, 17.317045211791992, 16.39027214050293, 15.136488914489746, 15.445602416992188, 17.8436279296875, 16.535795211791992, 15.139182090759277, 14.735954284667969, 14.4190673828125, 16.178855895996094, 15.463706970214844, 16.468944549560547, 16.784513473510742, 15.393339157104492, 16.55181312561035, 15.855633735656738, 16.35038948059082, 16.365015029907227, 16.792165756225586, 15.313777923583984, 15.428441047668457, 16.460540771484375, 16.795106887817383, 16.12445640563965, 16.688365936279297, 17.55154800415039]], dtype='float32').reshape([1, 30]),
            paddle.to_tensor(0, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b22560f93097aa986aaf12dfe5a985b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6d2aa12b1e7c95db4ba5ae21f76f2b4
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.9999998807907104, 0.9999998807907104, 0.9999998807907104, 0.9999998807907104]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor(0, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9bc638ea6914e9408065ef21d93fb188(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_935be8ec86ea0846f90144c0da32c940
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 30], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([[16.8125, 15.4296875, 16.765625, 17.15625, 16.265625, 15.4453125, 15.6875, 16.046875, 16.296875, 16.171875, 16.59375, 15.671875, 17.484375, 14.765625, 16.953125, 17.375, 17.46875, 15.53125, 16.34375, 16.5, 15.75, 16.609375, 15.1640625, 16.5625, 15.125, 16.703125, 16.03125, 16.5625, 16.265625, 17.203125]], dtype='float16').reshape([1, 30]),
            paddle.to_tensor(0, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_36d7fc9587fa4a4f55cc782abeb20d73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_935be8ec86ea0846f90144c0da32c940
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 4], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([[1.0, 1.0, 1.0, 1.0]], dtype='float16').reshape([1, 4]),
            paddle.to_tensor(0, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b13c73fdf7589178b0d4451679cbb1a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2_0, arg_3_0, arg_4):
        arg_2_0 = paddle._C_ops.full_int_array(0, paddle.int64, paddle.core.CPUPlace())
        arg_3_0 = paddle._C_ops.full_int_array(1, paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = [arg_2_0]
        input_3 = [arg_3_0]
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor(input_0, input_1, input_2, input_3, input_4, [1], [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 501, 256], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 256], dtype='float16'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a31e70f95d3915c6cf16c8eaadbfc1c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b13c73fdf7589178b0d4451679cbb1a6
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 256], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor(0, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a1413ad0334cdb18e10c09e936fe3be7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2_0, arg_3_0, arg_4):
        arg_2_0 = paddle._C_ops.full_int_array(0, paddle.int64, paddle.core.CPUPlace())
        arg_3_0 = paddle._C_ops.full_int_array(1, paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = [arg_2_0]
        input_3 = [arg_3_0]
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor(input_0, input_1, input_2, input_3, input_4, [1], [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 501, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3efaf32cd4bd39558a42a600ad97e405(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1413ad0334cdb18e10c09e936fe3be7
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(0, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4872d4d1e096af03b79834a4dd6dc908(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2_0, arg_3_0, arg_4):
        arg_2_0 = paddle._C_ops.full_int_array(0, paddle.int64, paddle.core.CPUPlace())
        arg_3_0 = paddle._C_ops.full_int_array(1, paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = [arg_2_0]
        input_3 = [arg_3_0]
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor(input_0, input_1, input_2, input_3, input_4, [1], [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 501, 30], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 30], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fef2345bd12427b0cde6ebe1af8d9841(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4872d4d1e096af03b79834a4dd6dc908
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 30], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[16.966392517089844, 16.338186264038086, 15.731739044189453, 15.963295936584473, 17.317045211791992, 16.39027214050293, 15.136488914489746, 15.445602416992188, 17.8436279296875, 16.535795211791992, 15.139182090759277, 14.735954284667969, 14.4190673828125, 16.178855895996094, 15.463706970214844, 16.468944549560547, 16.784513473510742, 15.393339157104492, 16.55181312561035, 15.855633735656738, 16.35038948059082, 16.365015029907227, 16.792165756225586, 15.313777923583984, 15.428441047668457, 16.460540771484375, 16.795106887817383, 16.12445640563965, 16.688365936279297, 17.55154800415039]], dtype='float32').reshape([1, 30]),
            paddle.to_tensor(0, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_55d01031b490d27f10d1c5e089d10a69(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2_0, arg_3_0, arg_4):
        arg_2_0 = paddle._C_ops.full_int_array(0, paddle.int64, paddle.core.CPUPlace())
        arg_3_0 = paddle._C_ops.full_int_array(1, paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = [arg_2_0]
        input_3 = [arg_3_0]
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor(input_0, input_1, input_2, input_3, input_4, [1], [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 501, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_02de54381fec9b5e51a54754dc60ccdf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55d01031b490d27f10d1c5e089d10a69
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0.9999998807907104, 0.9999998807907104, 0.9999998807907104, 0.9999998807907104]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor(0, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_45095ccde23c8ca759b8aed67aa3a1b2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2_0, arg_3_0, arg_4):
        arg_2_0 = paddle._C_ops.full_int_array(0, paddle.int64, paddle.core.CPUPlace())
        arg_3_0 = paddle._C_ops.full_int_array(1, paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = [arg_2_0]
        input_3 = [arg_3_0]
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor(input_0, input_1, input_2, input_3, input_4, [1], [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 501, 30], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 30], dtype='float16'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_98df2766eba9ccfee3c254c34c269b86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45095ccde23c8ca759b8aed67aa3a1b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 30], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([[16.8125, 15.4296875, 16.765625, 17.15625, 16.265625, 15.4453125, 15.6875, 16.046875, 16.296875, 16.171875, 16.59375, 15.671875, 17.484375, 14.765625, 16.953125, 17.375, 17.46875, 15.53125, 16.34375, 16.5, 15.75, 16.609375, 15.1640625, 16.5625, 15.125, 16.703125, 16.03125, 16.5625, 16.265625, 17.203125]], dtype='float16').reshape([1, 30]),
            paddle.to_tensor(0, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d562597ff5f872b53d27cddc427a7e71(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2_0, arg_3_0, arg_4):
        arg_2_0 = paddle._C_ops.full_int_array(0, paddle.int64, paddle.core.CPUPlace())
        arg_3_0 = paddle._C_ops.full_int_array(1, paddle.int64, paddle.core.CPUPlace())
        arg_4 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = [arg_2_0]
        input_3 = [arg_3_0]
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor(input_0, input_1, input_2, input_3, input_4, [1], [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 501, 4], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 4], dtype='float16'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3394da1f5f1ecda9126672111bb5a239(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d562597ff5f872b53d27cddc427a7e71
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 4], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([[1.0, 1.0, 1.0, 1.0]], dtype='float16').reshape([1, 4]),
            paddle.to_tensor(0, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


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