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
class PrimitiveOp_5b541091fef28549289db4d1de630688(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1, arg_1_2):
        arg_1_0 = paddle._C_ops.full_int_array([1], paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([3], paddle.int32, paddle.core.CPUPlace())
        arg_1_2 = paddle._C_ops.full_int_array([2], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1, arg_1_2]
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b422936dec3006cb0676be6fb2e34b18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b541091fef28549289db4d1de630688
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.056889720261096954, 0.4661372900009155], [0.09056175500154495, 0.4642089307308197], [0.4937588572502136, 0.4658372700214386]], dtype='float32').reshape([3, 2]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
            paddle.to_tensor([3], dtype='int32').reshape([1]),
            paddle.to_tensor([2], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_fcbdd3a57a7e1b2a3f608784808cfb68(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1, arg_1_2):
        arg_1_0 = paddle._C_ops.full_int_array([1], paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([-1], paddle.int32, paddle.core.CPUPlace())
        arg_1_2 = paddle._C_ops.full_int_array([-1], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1, arg_1_2]
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0e10fc54d10953aee8581878cde06295(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fcbdd3a57a7e1b2a3f608784808cfb68
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 384], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3d2793a011756639f0b0766b63e5113e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fcbdd3a57a7e1b2a3f608784808cfb68
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 768], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_47699f5c3bd48f35b25ce4450ab61c5f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1):
        arg_1_0 = paddle._C_ops.full_int_array([1], paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([26], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_60873bb8a1c5f830b527881f4c4e84fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_47699f5c3bd48f35b25ce4450ab61c5f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]], dtype='int64').reshape([1, 26]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
            paddle.to_tensor([26], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6a27006bb3a22c3743f4b457d98fbe1f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1):
        arg_1_0 = paddle._C_ops.full_int_array([100], paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([168], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_418789bd3a3c229865ce6498044ca238(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a27006bb3a22c3743f4b457d98fbe1f
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 168], dtype='int64'), 'int64'),
            paddle.to_tensor([100], dtype='int32').reshape([1]),
            paddle.to_tensor([168], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d5721a9a69c886e81c678fc37c8b555b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a27006bb3a22c3743f4b457d98fbe1f
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[100, 1], dtype='int64'), 'int64'),
            paddle.to_tensor([100], dtype='int32').reshape([1]),
            paddle.to_tensor([168], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b6003d23521c5a49ae8321bfaa3a7aec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1):
        arg_1_0 = paddle._C_ops.full_int_array([50], paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([84], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f529a28238e872fd934c7f57ba71fb29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6003d23521c5a49ae8321bfaa3a7aec
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 84], dtype='int64'), 'int64'),
            paddle.to_tensor([50], dtype='int32').reshape([1]),
            paddle.to_tensor([84], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_656854460f806849f330d8fa5eba31e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6003d23521c5a49ae8321bfaa3a7aec
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[50, 1], dtype='int64'), 'int64'),
            paddle.to_tensor([50], dtype='int32').reshape([1]),
            paddle.to_tensor([84], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ed173dc41bc3223f2d2ceddc7d77b133(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1):
        arg_1_0 = paddle._C_ops.full_int_array([25], paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([42], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d232b7035cac505b60050463f454d049(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ed173dc41bc3223f2d2ceddc7d77b133
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 42], dtype='int64'), 'int64'),
            paddle.to_tensor([25], dtype='int32').reshape([1]),
            paddle.to_tensor([42], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0079445fcf958b785e2fcc1105cb773f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ed173dc41bc3223f2d2ceddc7d77b133
    def get_inputs(self):
        return [
            paddle.to_tensor([[0], [32], [64], [96], [128], [160], [192], [224], [256], [288], [320], [352], [384], [416], [448], [480], [512], [544], [576], [608], [640], [672], [704], [736], [768]], dtype='int64').reshape([25, 1]),
            paddle.to_tensor([25], dtype='int32').reshape([1]),
            paddle.to_tensor([42], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e7ed5d345ad843a1a55a993dec5d9c05(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1):
        arg_1_0 = paddle._C_ops.full_int_array([13], paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([21], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4a40d5cf16f55b512dd3bd8558a01fea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7ed5d345ad843a1a55a993dec5d9c05
    def get_inputs(self):
        return [
            paddle.to_tensor([[0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280]], dtype='int64').reshape([1, 21]),
            paddle.to_tensor([13], dtype='int32').reshape([1]),
            paddle.to_tensor([21], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f3325b88ae6bb0edce10073a13d9ed93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7ed5d345ad843a1a55a993dec5d9c05
    def get_inputs(self):
        return [
            paddle.to_tensor([[0], [64], [128], [192], [256], [320], [384], [448], [512], [576], [640], [704], [768]], dtype='int64').reshape([13, 1]),
            paddle.to_tensor([13], dtype='int32').reshape([1]),
            paddle.to_tensor([21], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e3965cd19bf0563c5523b1f4b25978bf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1):
        arg_1_0 = paddle._C_ops.full_int_array([7], paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([11], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0af82e268fdd92543cff0955d72029fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3965cd19bf0563c5523b1f4b25978bf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0, 128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280]], dtype='int64').reshape([1, 11]),
            paddle.to_tensor([7], dtype='int32').reshape([1]),
            paddle.to_tensor([11], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_583475ec571da93611bc20403b0ed841(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3965cd19bf0563c5523b1f4b25978bf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0], [128], [256], [384], [512], [640], [768]], dtype='int64').reshape([7, 1]),
            paddle.to_tensor([7], dtype='int32').reshape([1]),
            paddle.to_tensor([11], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c2648c788c4cfaf6e7629e277e90e7c5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1, arg_1_2):
        arg_1_0 = paddle._C_ops.full_int_array([1], paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([-1], paddle.int32, paddle.core.CPUPlace())
        arg_1_2 = paddle._C_ops.full_int_array([-1], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1, arg_1_2]
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_90ea3a1ae67e290109cb600bd1d9144c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2648c788c4cfaf6e7629e277e90e7c5
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4c99d320cd27ad1fbd44309a4b97fbc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2648c788c4cfaf6e7629e277e90e7c5
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f701ec8a598a929d9057843424d9f7af(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 80, 80, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a343ea37d163ab32ea385b0b162dc227(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f701ec8a598a929d9057843424d9f7af
    def get_inputs(self):
        return [
            paddle.uniform([80, 80, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1, 3, 80, 80, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ffac6b51a91ac77a972489bd2dbf75cc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 80, 80, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_92014ab9aad19bd16eeca080aa97f63d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffac6b51a91ac77a972489bd2dbf75cc
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[[0.156005859375, 0.343505859375]]], [[[0.180908203125, 0.3818359375]]], [[[0.461181640625, 0.018585205078125]]]]], dtype='float16').reshape([1, 3, 1, 1, 2]),
            paddle.to_tensor([1, 3, 80, 80, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2c0bb69cd75aa2a0fb18c0f02fb9af6e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 40, 40, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1f05dcc6110708adc21a21dc26f70ab7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c0bb69cd75aa2a0fb18c0f02fb9af6e
    def get_inputs(self):
        return [
            paddle.uniform([40, 40, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1, 3, 40, 40, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_65e82737129f5ff1d5edec467c511c55(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 40, 40, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_40a3f4a08029b8bbd2e37b52aa81bf33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65e82737129f5ff1d5edec467c511c55
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[[0.26123046875, 0.3828125]]], [[[0.324462890625, 0.284423828125]]], [[[0.4833984375, 0.399169921875]]]]], dtype='float16').reshape([1, 3, 1, 1, 2]),
            paddle.to_tensor([1, 3, 40, 40, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9ad1122bd8e713189fdec2e0e12377c2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 20, 20, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_269cc63e2f72a7a395dbb30078221b59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9ad1122bd8e713189fdec2e0e12377c2
    def get_inputs(self):
        return [
            paddle.uniform([20, 20, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1, 3, 20, 20, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8eba4165a62fb3bb41200aba6f02de32(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 20, 20, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_14dda8fdf95f747f825aeb4f84ee0054(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8eba4165a62fb3bb41200aba6f02de32
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[[0.472412109375, 0.300537109375]]], [[[0.279296875, 0.283203125]]], [[[0.460693359375, 0.0540771484375]]]]], dtype='float16').reshape([1, 3, 1, 1, 2]),
            paddle.to_tensor([1, 3, 20, 20, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_aea5b74822a641a85c6867f77afb1b66(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 80, 80, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7e2859887940faa5feabe6e76caa2dc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aea5b74822a641a85c6867f77afb1b66
    def get_inputs(self):
        return [
            paddle.uniform([80, 80, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 3, 80, 80, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ee21004f7cca4e47442bb7ec98ee4585(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 80, 80, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a69a9420870412319c25e8f46cb435e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee21004f7cca4e47442bb7ec98ee4585
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[[0.12098323553800583, 0.23452886939048767]]], [[[0.4013497233390808, 0.08864551782608032]]], [[[0.062101978808641434, 0.0683494582772255]]]]], dtype='float32').reshape([1, 3, 1, 1, 2]),
            paddle.to_tensor([1, 3, 80, 80, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_49c335aee2f5e66b99c7439ccada3063(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 40, 40, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a6dc8b4aa79950087ff7a6c1e713afc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_49c335aee2f5e66b99c7439ccada3063
    def get_inputs(self):
        return [
            paddle.uniform([40, 40, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 3, 40, 40, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_3057e0955b51dcd688b2e0c49690fd9d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 40, 40, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_09f7bc91a2938312ee2dcbfbe1f72a97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3057e0955b51dcd688b2e0c49690fd9d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[[0.42240506410598755, 0.45154112577438354]]], [[[0.18279752135276794, 0.19247856736183167]]], [[[0.1981535404920578, 0.01877675950527191]]]]], dtype='float32').reshape([1, 3, 1, 1, 2]),
            paddle.to_tensor([1, 3, 40, 40, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5f8e349549466c8fbac20c02609943f3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 20, 20, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9131fc61bc15e9f69b8d4d40a9ff428f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f8e349549466c8fbac20c02609943f3
    def get_inputs(self):
        return [
            paddle.uniform([20, 20, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 3, 20, 20, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_375e542547f226333b7f36e4c07566a8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 20, 20, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f53dabb4b904384d809ebe8b33bf9fc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_375e542547f226333b7f36e4c07566a8
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[[0.27094730734825134, 0.09780169278383255]]], [[[0.3459741473197937, 0.4000910222530365]]], [[[0.08794714510440826, 0.15952467918395996]]]]], dtype='float32').reshape([1, 3, 1, 1, 2]),
            paddle.to_tensor([1, 3, 20, 20, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_10b49c73a079c591032bdde7f98d596d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1, arg_1_2):
        arg_1_0 = paddle._C_ops.full_int_array([1], paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([3], paddle.int32, paddle.core.CPUPlace())
        arg_1_2 = paddle._C_ops.full_int_array([2], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1, arg_1_2]
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_08cf0cf41ad0146e5bf9d8882c35f307(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10b49c73a079c591032bdde7f98d596d
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.314697265625, 0.3466796875], [0.3134765625, 0.361328125], [0.044158935546875, 0.2171630859375]], dtype='float16').reshape([3, 2]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
            paddle.to_tensor([3], dtype='int32').reshape([1]),
            paddle.to_tensor([2], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1de901f2f39a12b65afc4ae37973ef86(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1, arg_1_2):
        arg_1_0 = paddle._C_ops.full_int_array([1], paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([3], paddle.int32, paddle.core.CPUPlace())
        arg_1_2 = paddle._C_ops.full_int_array([2], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1, arg_1_2]
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c6f19d5504d7dd89b68e9d31c6cca79a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1de901f2f39a12b65afc4ae37973ef86
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.056889720261096954, 0.4661372900009155], [0.09056175500154495, 0.4642089307308197], [0.4937588572502136, 0.4658372700214386]], dtype='float32').reshape([3, 2]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
            paddle.to_tensor([3], dtype='int32').reshape([1]),
            paddle.to_tensor([2], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8f6a464f71ea18da1c90a4082b3460b6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1, arg_1_2):
        arg_1_0 = paddle._C_ops.full_int_array([1], paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([-1], paddle.int32, paddle.core.CPUPlace())
        arg_1_2 = paddle._C_ops.full_int_array([-1], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1, arg_1_2]
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 384], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_db64bf7a9dd02408e5551b7d209bc58b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f6a464f71ea18da1c90a4082b3460b6
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 384], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_439622618f0adbc439eb8c87dee4b578(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1, arg_1_2):
        arg_1_0 = paddle._C_ops.full_int_array([1], paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([-1], paddle.int32, paddle.core.CPUPlace())
        arg_1_2 = paddle._C_ops.full_int_array([-1], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1, arg_1_2]
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 768], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1b664dbf77a49b6df58ef28d1f043149(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_439622618f0adbc439eb8c87dee4b578
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 768], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8ed8c9cc29160796db2390feb4f8dede(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1):
        arg_1_0 = paddle._C_ops.full_int_array([1], paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([26], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 26], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_034d1c0f6b3776238e9a5f1548b70de3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ed8c9cc29160796db2390feb4f8dede
    def get_inputs(self):
        return [
            paddle.to_tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]], dtype='int64').reshape([1, 26]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
            paddle.to_tensor([26], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_886c9e2725a76fef1fec7d34e869e935(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1):
        arg_1_0 = paddle._C_ops.full_int_array([100], paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([168], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, None], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0d03038da0fe005a0c1ba6dd448840fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_886c9e2725a76fef1fec7d34e869e935
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 168], dtype='int64'), 'int64'),
            paddle.to_tensor([100], dtype='int32').reshape([1]),
            paddle.to_tensor([168], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c7b73700b78dce6f3b2ea708acdbff9b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1):
        arg_1_0 = paddle._C_ops.full_int_array([100], paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([168], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1cd4fafb92a9e1d33af1913fd54c9a43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7b73700b78dce6f3b2ea708acdbff9b
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[100, 1], dtype='int64'), 'int64'),
            paddle.to_tensor([100], dtype='int32').reshape([1]),
            paddle.to_tensor([168], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_bdad975873108fa62a54336c4278479e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1):
        arg_1_0 = paddle._C_ops.full_int_array([50], paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([84], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, None], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c36bf18b281b665e170eac194804751c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bdad975873108fa62a54336c4278479e
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 84], dtype='int64'), 'int64'),
            paddle.to_tensor([50], dtype='int32').reshape([1]),
            paddle.to_tensor([84], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_607ad8a4fe738b3de7a50f88ce1ba9f2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1):
        arg_1_0 = paddle._C_ops.full_int_array([50], paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([84], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c1c811d10be52ac39f163ea5c21b2e31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_607ad8a4fe738b3de7a50f88ce1ba9f2
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[50, 1], dtype='int64'), 'int64'),
            paddle.to_tensor([50], dtype='int32').reshape([1]),
            paddle.to_tensor([84], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_acf9e8ba7a507bd106f847c48c759fba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1):
        arg_1_0 = paddle._C_ops.full_int_array([25], paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([42], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, None], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8dda8f359bd2d2de7fbb399bfe42cda5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_acf9e8ba7a507bd106f847c48c759fba
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 42], dtype='int64'), 'int64'),
            paddle.to_tensor([25], dtype='int32').reshape([1]),
            paddle.to_tensor([42], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b27a8fb1448b0241733942cbb9549773(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1):
        arg_1_0 = paddle._C_ops.full_int_array([25], paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([42], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d88a98127a74d43b0af427161e24bee6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b27a8fb1448b0241733942cbb9549773
    def get_inputs(self):
        return [
            paddle.to_tensor([[0], [32], [64], [96], [128], [160], [192], [224], [256], [288], [320], [352], [384], [416], [448], [480], [512], [544], [576], [608], [640], [672], [704], [736], [768]], dtype='int64').reshape([25, 1]),
            paddle.to_tensor([25], dtype='int32').reshape([1]),
            paddle.to_tensor([42], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_bbc5b89237ca4f27d5e9b29f1889f2a8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1):
        arg_1_0 = paddle._C_ops.full_int_array([13], paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([21], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, None], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_51fda02a8386f66aa933b2923eb3c76e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbc5b89237ca4f27d5e9b29f1889f2a8
    def get_inputs(self):
        return [
            paddle.to_tensor([[0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280]], dtype='int64').reshape([1, 21]),
            paddle.to_tensor([13], dtype='int32').reshape([1]),
            paddle.to_tensor([21], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_57ae1dbab506ee9916b336d90284f01a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1):
        arg_1_0 = paddle._C_ops.full_int_array([13], paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([21], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_06235c85ee3f5f4a4bc1c434488e4ad5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57ae1dbab506ee9916b336d90284f01a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0], [64], [128], [192], [256], [320], [384], [448], [512], [576], [640], [704], [768]], dtype='int64').reshape([13, 1]),
            paddle.to_tensor([13], dtype='int32').reshape([1]),
            paddle.to_tensor([21], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_cbf43f50022d5c11479614bf87b27024(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1):
        arg_1_0 = paddle._C_ops.full_int_array([7], paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([11], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, None], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e9a17e8d769a85428321456e46b2d0e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbf43f50022d5c11479614bf87b27024
    def get_inputs(self):
        return [
            paddle.to_tensor([[0, 128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280]], dtype='int64').reshape([1, 11]),
            paddle.to_tensor([7], dtype='int32').reshape([1]),
            paddle.to_tensor([11], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a3837b83948071420c20f9ed464d29c0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1):
        arg_1_0 = paddle._C_ops.full_int_array([7], paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([11], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_53042d8812d3d4e9bbeeebc5c112ea9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3837b83948071420c20f9ed464d29c0
    def get_inputs(self):
        return [
            paddle.to_tensor([[0], [128], [256], [384], [512], [640], [768]], dtype='int64').reshape([7, 1]),
            paddle.to_tensor([7], dtype='int32').reshape([1]),
            paddle.to_tensor([11], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e2698d673edda55bbf25f657708c43e9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1, arg_1_2):
        arg_1_0 = paddle._C_ops.full_int_array([1], paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([-1], paddle.int32, paddle.core.CPUPlace())
        arg_1_2 = paddle._C_ops.full_int_array([-1], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1, arg_1_2]
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bfdfcb7d22049c52e95ad348491443fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2698d673edda55bbf25f657708c43e9
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ccb138510682b3ff8a3fc0ab36bcfc2f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1, arg_1_2):
        arg_1_0 = paddle._C_ops.full_int_array([1], paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([-1], paddle.int32, paddle.core.CPUPlace())
        arg_1_2 = paddle._C_ops.full_int_array([-1], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1, arg_1_2]
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_229e8dc79b8fde9fb4e5b6012ca0fc93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccb138510682b3ff8a3fc0ab36bcfc2f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 768], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
            paddle.to_tensor([-1], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6bdce6b84bfd67002cf04115b98a5369(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 80, 80, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[80, 80, 2], dtype='float16'),
            paddle.static.InputSpec(shape=[5], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_31180049b1595e94f0319f5afc7332ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bdce6b84bfd67002cf04115b98a5369
    def get_inputs(self):
        return [
            paddle.uniform([80, 80, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1, 3, 80, 80, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_60bdd1c1ad4a1da7abcb6ae8962feb89(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 80, 80, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 1, 1, 2], dtype='float16'),
            paddle.static.InputSpec(shape=[5], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dd5a83e2b9a8a6362925b449a6e68c07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60bdd1c1ad4a1da7abcb6ae8962feb89
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[[0.156005859375, 0.343505859375]]], [[[0.180908203125, 0.3818359375]]], [[[0.461181640625, 0.018585205078125]]]]], dtype='float16').reshape([1, 3, 1, 1, 2]),
            paddle.to_tensor([1, 3, 80, 80, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_cc3a7fa1f6597a9d865009233b5588b5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 40, 40, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[40, 40, 2], dtype='float16'),
            paddle.static.InputSpec(shape=[5], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_559bd8186c2182c9ed52f90a28a3b19b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc3a7fa1f6597a9d865009233b5588b5
    def get_inputs(self):
        return [
            paddle.uniform([40, 40, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1, 3, 40, 40, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2a85327b217add3a90bfccdb1e71f0f7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 40, 40, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 1, 1, 2], dtype='float16'),
            paddle.static.InputSpec(shape=[5], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_52226105d5f026c2c8431ebfbc02552b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a85327b217add3a90bfccdb1e71f0f7
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[[0.26123046875, 0.3828125]]], [[[0.324462890625, 0.284423828125]]], [[[0.4833984375, 0.399169921875]]]]], dtype='float16').reshape([1, 3, 1, 1, 2]),
            paddle.to_tensor([1, 3, 40, 40, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1c42319fbe57fddaf3568ae679d76a8a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 20, 20, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20, 20, 2], dtype='float16'),
            paddle.static.InputSpec(shape=[5], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1dcefd634df17f6e347548385c5a9fa3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c42319fbe57fddaf3568ae679d76a8a
    def get_inputs(self):
        return [
            paddle.uniform([20, 20, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1, 3, 20, 20, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9783df7d56d0d4bdb33e509130bb1b73(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 20, 20, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 1, 1, 2], dtype='float16'),
            paddle.static.InputSpec(shape=[5], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d1ccefa07a26c91140ba8d9e4d4659ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9783df7d56d0d4bdb33e509130bb1b73
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[[0.472412109375, 0.300537109375]]], [[[0.279296875, 0.283203125]]], [[[0.460693359375, 0.0540771484375]]]]], dtype='float16').reshape([1, 3, 1, 1, 2]),
            paddle.to_tensor([1, 3, 20, 20, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4a81d798c54b24f330fc800fee557d31(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 80, 80, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[80, 80, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[5], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_111aa266cc5dd64605acb7846be2d9ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a81d798c54b24f330fc800fee557d31
    def get_inputs(self):
        return [
            paddle.uniform([80, 80, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 3, 80, 80, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_51a703effd6103998d40491866e9d675(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 80, 80, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 1, 1, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[5], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7179b89b867ded1aa68aeef9adef6d8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51a703effd6103998d40491866e9d675
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[[0.12098323553800583, 0.23452886939048767]]], [[[0.4013497233390808, 0.08864551782608032]]], [[[0.062101978808641434, 0.0683494582772255]]]]], dtype='float32').reshape([1, 3, 1, 1, 2]),
            paddle.to_tensor([1, 3, 80, 80, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d9ff143e88931cc7fb0b952581ff1de6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 40, 40, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[40, 40, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[5], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_922275da1e4ae3a338403d2854a843c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9ff143e88931cc7fb0b952581ff1de6
    def get_inputs(self):
        return [
            paddle.uniform([40, 40, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 3, 40, 40, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_281a3dea7e0f1eb8358122ab9b25d10e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 40, 40, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 1, 1, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[5], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_496f7505e17d33669b6accef766cc085(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_281a3dea7e0f1eb8358122ab9b25d10e
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[[0.42240506410598755, 0.45154112577438354]]], [[[0.18279752135276794, 0.19247856736183167]]], [[[0.1981535404920578, 0.01877675950527191]]]]], dtype='float32').reshape([1, 3, 1, 1, 2]),
            paddle.to_tensor([1, 3, 40, 40, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_da75383804267717813edb3dc1903e62(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 20, 20, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20, 20, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[5], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_215864605da4de03c1e3d595b915be2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da75383804267717813edb3dc1903e62
    def get_inputs(self):
        return [
            paddle.uniform([20, 20, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 3, 20, 20, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_62b8c045348ee71c475646ecabe0a43c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 3, 20, 20, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 1, 1, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[5], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0215614af3b8ec19661b8dce86437d96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62b8c045348ee71c475646ecabe0a43c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[[0.27094730734825134, 0.09780169278383255]]], [[[0.3459741473197937, 0.4000910222530365]]], [[[0.08794714510440826, 0.15952467918395996]]]]], dtype='float32').reshape([1, 3, 1, 1, 2]),
            paddle.to_tensor([1, 3, 20, 20, 2], dtype='int64').reshape([5]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9aee7fb2be38074c3541d628b5323383(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1, arg_1_2):
        arg_1_0 = paddle._C_ops.full_int_array([1], paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([3], paddle.int32, paddle.core.CPUPlace())
        arg_1_2 = paddle._C_ops.full_int_array([2], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1, arg_1_2]
        return paddle._C_ops.expand(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3, 2], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d5d96090e9e43f02f1cfcb9f6e4239d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9aee7fb2be38074c3541d628b5323383
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.314697265625, 0.3466796875], [0.3134765625, 0.361328125], [0.044158935546875, 0.2171630859375]], dtype='float16').reshape([3, 2]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
            paddle.to_tensor([3], dtype='int32').reshape([1]),
            paddle.to_tensor([2], dtype='int32').reshape([1]),
        ]


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