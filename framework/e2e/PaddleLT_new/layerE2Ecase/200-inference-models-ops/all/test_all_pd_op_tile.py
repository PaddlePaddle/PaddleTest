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
class PrimitiveOp_59727dc15a15eef270324e0075d4cc3b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1):
        arg_1_0 = paddle._C_ops.full_int_array(1, paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([1], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d2ecf504af7a5d7604f660daf6143a5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59727dc15a15eef270324e0075d4cc3b
    def get_inputs(self):
        return [
            paddle.to_tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]], dtype='int64').reshape([1, 26]),
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9de0b7f0665f6a476cc68b7271da8a51(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 20, 1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f9bdb5bfab46a6e56b9dd98a94de258f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9de0b7f0665f6a476cc68b7271da8a51
    def get_inputs(self):
        return [
            paddle.uniform([3200, 1, 2], dtype='float64', min=0, max=0.5),
            paddle.to_tensor([1, 20, 1], dtype='int64').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2f4314477bef49e304b5677c578f0873(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0):
        arg_1_0 = paddle._C_ops.full_int_array(128, paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_23f218bd5636e0102989e0d392271f48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f4314477bef49e304b5677c578f0873
    def get_inputs(self):
        return [
            paddle.uniform([128, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1c4e6485aa071b03f4a88a8dac49134d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0):
        arg_1_0 = paddle._C_ops.full_int_array(256, paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d915d274274ad766b6c9ef870cbcb140(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c4e6485aa071b03f4a88a8dac49134d
    def get_inputs(self):
        return [
            paddle.uniform([256, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(256, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0847060dbbc7895518cf85462c75e5b4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 72, 1, 1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a66cdf4e8acbfeb9f19acc1b8a0ceb79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0847060dbbc7895518cf85462c75e5b4
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 180, 320], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1, 72, 1, 1], dtype='int64').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6028e25b2476888a5ade6ef366b27e82(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 36, 1, 1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1f82ab04aae4e66452b135cc23109812(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6028e25b2476888a5ade6ef366b27e82
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 180, 320], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1, 36, 1, 1], dtype='int64').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e4c1848561ec02f18f92cd98f6b04286(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0):
        arg_1_0 = paddle._C_ops.full_int_array(128, paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_86ce6f15be5bce1e539914a3dd3da60b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4c1848561ec02f18f92cd98f6b04286
    def get_inputs(self):
        return [
            paddle.uniform([128, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_91650acbb60a690e3fa6a9e950a565cf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0):
        arg_1_0 = paddle._C_ops.full_int_array(256, paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e57eebbed3249719b175ffa4e4a921d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91650acbb60a690e3fa6a9e950a565cf
    def get_inputs(self):
        return [
            paddle.uniform([256, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor(256, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_96ec2be7d6a8954a86e0e89142c0f78b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f1f2341887ca5106c3c16710224c2a79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96ec2be7d6a8954a86e0e89142c0f78b
    def get_inputs(self):
        return [
            paddle.to_tensor([[1.0, 1.0]], dtype='float16').reshape([1, 2]),
            paddle.to_tensor([1, 2], dtype='int64').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_538d3c4e56389394c6c9bdc8e5730584(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3357d5978e62a973989f500040e7992b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_538d3c4e56389394c6c9bdc8e5730584
    def get_inputs(self):
        return [
            paddle.to_tensor([[1.0, 1.0]], dtype='float32').reshape([1, 2]),
            paddle.to_tensor([1, 2], dtype='int64').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b81b33a5d58291b0951202335ee1a412(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 72, 1, 1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d67e30a76f1220fa272f9bcb85e22693(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b81b33a5d58291b0951202335ee1a412
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 72, 1, 1], dtype='int64').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_703ca1ea11c802ec0b25f2c11a52ceae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 36, 1, 1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fc62d33d198db72f56934015f5bcc73d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_703ca1ea11c802ec0b25f2c11a52ceae
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 36, 1, 1], dtype='int64').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d42576d945732d6046d855c671d20d5b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0, arg_1_1):
        arg_1_0 = paddle._C_ops.full_int_array(1, paddle.int32, paddle.core.CPUPlace())
        arg_1_1 = paddle._C_ops.full_int_array([1], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0, arg_1_1]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 26], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cccd0ce91f20b37be20c1f93e333a04f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d42576d945732d6046d855c671d20d5b
    def get_inputs(self):
        return [
            paddle.to_tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]], dtype='int64').reshape([1, 26]),
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1fd22e5c75b4b2e4ad159938a4955a5e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 20, 1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3200, 1, 2], dtype='float64'),
            paddle.static.InputSpec(shape=[3], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_55ba1c073ef5a26d88d97c70a01758dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fd22e5c75b4b2e4ad159938a4955a5e
    def get_inputs(self):
        return [
            paddle.uniform([3200, 1, 2], dtype='float64', min=0, max=0.5),
            paddle.to_tensor([1, 20, 1], dtype='int64').reshape([3]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2fad7fc0b907119e24621ad3ffb185df(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0):
        arg_1_0 = paddle._C_ops.full_int_array(128, paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_351abbce1bc204f698bf2d6e4468a042(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2fad7fc0b907119e24621ad3ffb185df
    def get_inputs(self):
        return [
            paddle.uniform([128, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_3898a5210568a12fb09bafd7f62db946(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0):
        arg_1_0 = paddle._C_ops.full_int_array(256, paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5a8213fe67b83e7b17a81b529631026a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3898a5210568a12fb09bafd7f62db946
    def get_inputs(self):
        return [
            paddle.uniform([256, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(256, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b8cf409859e99131981849a05da191c0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 72, 1, 1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 180, 320], dtype='float16'),
            paddle.static.InputSpec(shape=[4], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a69a322747bb0009b46012239bbbc71a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8cf409859e99131981849a05da191c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 180, 320], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1, 72, 1, 1], dtype='int64').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d0208ef17c143acc71353f9bad51f6cb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 36, 1, 1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 180, 320], dtype='float16'),
            paddle.static.InputSpec(shape=[4], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_feb0547f2dc72692308e23f3c982b0ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d0208ef17c143acc71353f9bad51f6cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 180, 320], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1, 36, 1, 1], dtype='int64').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_901d393af54ff625b1bfe9c43881ca4a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0):
        arg_1_0 = paddle._C_ops.full_int_array(128, paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7cb775ed9071bf0d75fc4e2b1ce638b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_901d393af54ff625b1bfe9c43881ca4a
    def get_inputs(self):
        return [
            paddle.uniform([128, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor(128, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5a5dfa0effa53bcef7f140b4d27f5978(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1_0):
        arg_1_0 = paddle._C_ops.full_int_array(256, paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = [arg_1_0]
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_952b1763e24e59156f0ab0ebf5964a40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a5dfa0effa53bcef7f140b4d27f5978
    def get_inputs(self):
        return [
            paddle.uniform([256, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor(256, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ada747b36b91a836711658685478798d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4f15dcbaceebbe45e7aaaaf1ec9724c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ada747b36b91a836711658685478798d
    def get_inputs(self):
        return [
            paddle.to_tensor([[1.0, 1.0]], dtype='float16').reshape([1, 2]),
            paddle.to_tensor([1, 2], dtype='int64').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6e002029a76fff49de87a4b5ef264773(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eff9e485e88153df09ad53838221a62c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e002029a76fff49de87a4b5ef264773
    def get_inputs(self):
        return [
            paddle.to_tensor([[1.0, 1.0]], dtype='float32').reshape([1, 2]),
            paddle.to_tensor([1, 2], dtype='int64').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4bb7717e4bee04fe2ed625b433533b20(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 72, 1, 1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 180, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[4], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2a1c0424da451dbe257ccbb21e5e82b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bb7717e4bee04fe2ed625b433533b20
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 72, 1, 1], dtype='int64').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c49e773253b85644453e0083568b3450(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 36, 1, 1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.tile(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 180, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[4], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4e562f8573700c9d1c954c1a10e5982f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c49e773253b85644453e0083568b3450
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 36, 1, 1], dtype='int64').reshape([4]),
        ]


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