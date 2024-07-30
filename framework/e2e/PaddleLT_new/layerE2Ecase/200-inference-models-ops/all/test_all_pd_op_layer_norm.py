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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_81d2664aee4a61236284940b97246272(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 512], dtype='float16', min=0, max=0.5),
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c997473e65a1f02cf3af8f02b2c327d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.12419048696756363, 0.37535029649734497, 0.2765030562877655, 0.3821343183517456, 0.33024221658706665, 0.07229235023260117, 0.45719510316848755, 0.3944741487503052, 0.20532748103141785, 0.05674845352768898, 0.12264791131019592, 0.3296707570552826, 0.40370532870292664, 0.21281811594963074, 0.3262869417667389, 0.01108761690557003, 0.4206697940826416, 0.46054691076278687, 0.27791014313697815, 0.1937817484140396, 0.19722622632980347, 0.08373107016086578, 0.16766349971294403, 0.2599410116672516], dtype='float32').reshape([24]),
            paddle.to_tensor([0.47431719303131104, 0.13286490738391876, 0.3111533522605896, 0.36972784996032715, 0.4015478491783142, 0.36506417393684387, 0.38756969571113586, 0.1611185073852539, 0.1466299444437027, 0.22819964587688446, 0.2328542172908783, 0.3959265947341919, 0.05855312570929527, 0.2452484518289566, 0.28116732835769653, 0.03424753248691559, 0.23618049919605255, 0.3465275168418884, 0.3895937204360962, 0.1697375327348709, 0.3248807489871979, 0.4428333640098572, 0.28583231568336487, 0.08076059073209763], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_136ef360122b641b9b3969f464d5ddcc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.029846541583538055, 0.4518411159515381, 0.1716068685054779, 0.2362489104270935, 0.3543258011341095, 0.40037932991981506, 0.23541198670864105, 0.167933389544487, 0.03557004779577255, 0.19448260962963104, 0.41416239738464355, 0.3681526482105255, 0.24877628684043884, 0.36061155796051025, 0.3229597210884094, 0.061476729810237885, 0.13109542429447174, 0.12173473089933395, 0.47740957140922546, 0.1820772886276245, 0.3760887086391449, 0.361057847738266, 0.2961604595184326, 0.34885305166244507], dtype='float32').reshape([24]),
            paddle.to_tensor([0.19965840876102448, 0.1485452502965927, 0.39907726645469666, 0.4856626093387604, 0.20386353135108948, 0.47867971658706665, 0.4497002363204956, 0.23218807578086853, 0.21636004745960236, 0.21615320444107056, 0.03118315525352955, 0.1251886934041977, 0.4594854712486267, 0.09600412845611572, 0.12832793593406677, 0.2248455286026001, 0.4104636013507843, 0.42970868945121765, 0.13298679888248444, 0.10439467430114746, 0.232858344912529, 0.40402474999427795, 0.4911373257637024, 0.23328770697116852], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b378706ea5933d43037bb9a4a78bc5b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.06122194975614548, 0.02137068659067154, 0.14039559662342072, 0.04391301050782204, 0.31955987215042114, 0.13062553107738495, 0.4961921274662018, 0.14633387327194214, 0.48925697803497314, 0.263266384601593, 0.261473685503006, 0.25453272461891174, 0.20250068604946136, 0.3981817960739136, 0.24953901767730713, 0.42823174595832825, 0.13289561867713928, 0.09234563261270523, 0.49178290367126465, 0.11917651444673538, 0.4463011920452118, 0.11164161562919617, 0.1418273001909256, 0.40878525376319885], dtype='float32').reshape([24]),
            paddle.to_tensor([0.47853320837020874, 0.471259206533432, 0.483436644077301, 0.15336103737354279, 0.22569091618061066, 0.3043261468410492, 0.43447503447532654, 0.2452462911605835, 0.49709680676460266, 0.4487230181694031, 0.425223708152771, 0.3794567883014679, 0.3276847004890442, 0.28613367676734924, 0.4307856261730194, 0.2105758637189865, 0.2616255581378937, 0.4875328540802002, 0.17232069373130798, 0.1616373360157013, 0.29326510429382324, 0.35090410709381104, 0.3807741105556488, 0.015944497659802437], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fe1ad41222677db5671cf502d54f2f41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4213341772556305, 0.3540722131729126, 0.05336308851838112, 0.47785329818725586, 0.29147499799728394, 0.3708406090736389, 0.4891560673713684, 0.329334020614624, 0.44010481238365173, 0.01798633672297001, 0.02813560888171196, 0.15297915041446686, 0.4731057286262512, 0.02022893913090229, 0.43038150668144226, 0.05641293525695801, 0.05831751227378845, 0.46261438727378845, 0.35460519790649414, 0.170273557305336, 0.15965817868709564, 0.11220718920230865, 0.339863657951355, 0.26430243253707886], dtype='float32').reshape([24]),
            paddle.to_tensor([0.24000950157642365, 0.2011338770389557, 0.3849775791168213, 0.1800290048122406, 0.050875041633844376, 0.04831603541970253, 0.37717726826667786, 0.1439453512430191, 0.025947287678718567, 0.44146913290023804, 0.3924444317817688, 0.06777292490005493, 0.11720570176839828, 0.3823343515396118, 0.34130412340164185, 0.4829157888889313, 0.08625774085521698, 0.04810171201825142, 0.0038659004494547844, 0.2229214608669281, 0.04871803894639015, 0.06826524436473846, 0.00794372707605362, 0.05092710256576538], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_08e5d9d1954f3d4e16c6ddbe64d7d883(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2705211341381073, 0.315491259098053, 0.02433658577501774, 0.3883836269378662, 0.1275789588689804, 0.1928188055753708, 0.37481293082237244, 0.2850351631641388, 0.060913003981113434, 0.11846338957548141, 0.05339489504694939, 0.32055607438087463, 0.3577638864517212, 0.19332900643348694, 0.3850342929363251, 0.3389863669872284, 0.4675685167312622, 0.1080484688282013, 0.38795146346092224, 0.3727991282939911, 0.46291935443878174, 0.49592286348342896, 0.16096989810466766, 0.2642364799976349], dtype='float32').reshape([24]),
            paddle.to_tensor([0.26355230808258057, 0.21223752200603485, 0.3469664752483368, 0.35968858003616333, 0.18014700710773468, 0.46183931827545166, 0.0240391343832016, 0.03798052296042442, 0.19690252840518951, 0.23696187138557434, 0.46961963176727295, 0.3380269408226013, 0.13394588232040405, 0.434874564409256, 0.14233030378818512, 0.0330117866396904, 0.31378623843193054, 0.42210060358047485, 0.018485013395547867, 0.4958232343196869, 0.12212290614843369, 0.4715007245540619, 0.21444234251976013, 0.37514764070510864], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a691ac7707602a5550ae8d62615d1869(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.08258425444364548, 0.0734853520989418, 0.09411701560020447, 0.05579982325434685, 0.06133255735039711, 0.39629992842674255, 0.4806704521179199, 0.4823877215385437, 0.12186381220817566, 0.4359857738018036, 0.23751281201839447, 0.30263516306877136, 0.30346933007240295, 0.2507888972759247, 0.03573431447148323, 0.27259814739227295, 0.19633232057094574, 0.1154838502407074, 0.26075252890586853, 0.16868750751018524, 0.053940530866384506, 0.29893261194229126, 0.19299538433551788, 0.17447715997695923], dtype='float32').reshape([24]),
            paddle.to_tensor([0.4466041922569275, 0.17117729783058167, 0.0720098540186882, 0.40316107869148254, 0.10955902189016342, 0.03818371146917343, 0.03343240171670914, 0.2106536626815796, 0.07636027038097382, 0.36251890659332275, 0.03222961351275444, 0.12276952713727951, 0.21909861266613007, 0.2954011857509613, 0.15671183168888092, 0.23567405343055725, 0.4272143542766571, 0.4417370855808258, 0.03791682422161102, 0.15880565345287323, 0.21285854279994965, 0.48467591404914856, 0.4582032561302185, 0.44239532947540283], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_65e5bdba527e75a1df4a7d41e48fe16d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.02179430052638054, 0.35659918189048767, 0.10288295149803162, 0.367136150598526, 0.03861471638083458, 0.3303942382335663, 0.02263519912958145, 0.08006787300109863, 0.39125028252601624, 0.07442966848611832, 0.4011555016040802, 0.3440481126308441, 0.4099203050136566, 0.08449126034975052, 0.2607165575027466, 0.46883636713027954, 0.010827611200511456, 0.47674131393432617, 0.01961446925997734, 0.23594188690185547, 0.4052788019180298, 0.062286511063575745, 0.12254875153303146, 0.15587499737739563], dtype='float32').reshape([24]),
            paddle.to_tensor([0.048090286552906036, 0.04565330594778061, 0.2716093063354492, 0.23335745930671692, 0.007170975673943758, 0.20589829981327057, 0.1202099472284317, 0.09331848472356796, 0.22194841504096985, 0.4402906894683838, 0.21088039875030518, 0.2652601897716522, 0.37642964720726013, 0.2923552989959717, 0.4936602711677551, 0.1778169572353363, 0.1487603783607483, 0.3199818730354309, 0.2537340223789215, 0.18421567976474762, 0.33916833996772766, 0.2629408538341522, 0.23394381999969482, 0.13371197879314423], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0e40e0a65e4a331ae12b973952afdcff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4390544295310974, 0.19970671832561493, 0.2590944766998291, 0.07582758367061615, 0.34973135590553284, 0.14893846213817596, 0.1972925364971161, 0.10873378068208694, 0.4033542275428772, 0.4635089039802551, 0.039704035967588425, 0.2580186724662781, 0.30740436911582947, 0.07510010153055191, 0.43295150995254517, 0.2231997400522232, 0.2723812460899353, 0.467327356338501, 0.37129393219947815, 0.21167562901973724, 0.37400370836257935, 0.4357066750526428, 0.4713532030582428, 0.2917383909225464], dtype='float32').reshape([24]),
            paddle.to_tensor([0.1983165144920349, 0.3643958270549774, 0.26831236481666565, 0.2816506028175354, 0.1044212281703949, 0.17534904181957245, 0.45651739835739136, 0.20121921598911285, 0.11607038974761963, 0.4561237394809723, 0.23605439066886902, 0.12967082858085632, 0.295977383852005, 0.37622329592704773, 0.21720823645591736, 0.2169981151819229, 0.1342230588197708, 0.21265734732151031, 0.4741648733615875, 0.22956861555576324, 0.14797234535217285, 0.36484959721565247, 0.31306618452072144, 0.08679403364658356], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2bf2be7a8f8c6176680b0d2cec1a6051(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.28779855370521545, 0.29507631063461304, 0.08921849727630615, 0.4362853765487671, 0.1639026254415512, 0.12181118130683899, 0.0925721600651741, 0.3030030429363251, 0.4857131540775299, 0.016851555556058884, 0.4665258228778839, 0.1737879514694214, 0.43905505537986755, 0.36795371770858765, 0.27491095662117004, 0.4894116222858429, 0.27295732498168945, 0.295064240694046, 0.20313318073749542, 0.05314280837774277, 0.07289009541273117, 0.37172722816467285, 0.20052076876163483, 0.2198108732700348], dtype='float32').reshape([24]),
            paddle.to_tensor([0.19671781361103058, 0.10109605640172958, 0.04864858463406563, 0.09422617405653, 0.3193044662475586, 0.4781797230243683, 0.28611403703689575, 0.25401005148887634, 0.10111551731824875, 0.017606079578399658, 0.04949459433555603, 0.16216453909873962, 0.15200424194335938, 0.24645820260047913, 0.21513457596302032, 0.10045173019170761, 0.3924245834350586, 0.37487998604774475, 0.03570882976055145, 0.015446366742253304, 0.033180106431245804, 0.2750890851020813, 0.17604152858257294, 0.2768828272819519], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2766a7c95f3f91699c165c6b4edb79c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4905850291252136, 0.1417711228132248, 0.3972661793231964, 0.26160258054733276, 0.2209509015083313, 0.040283795446157455, 0.2229292243719101, 0.365222692489624, 0.0916929617524147, 0.36036568880081177, 0.30587178468704224, 0.31058618426322937, 0.10337479412555695, 0.29634565114974976, 0.0766269713640213, 0.01825917512178421, 0.02897103875875473, 0.31862297654151917, 0.320491760969162, 0.23870548605918884, 0.1693069189786911, 0.004184379708021879, 0.29679298400878906, 0.03416705131530762], dtype='float32').reshape([24]),
            paddle.to_tensor([0.35126546025276184, 0.1637580841779709, 0.11904033273458481, 0.22047705948352814, 0.18035081028938293, 0.29125672578811646, 0.15367192029953003, 0.29683148860931396, 0.3086322546005249, 0.2522909939289093, 0.2270323932170868, 0.13263821601867676, 0.026715775951743126, 0.2140287458896637, 0.19878646731376648, 0.35739991068840027, 0.15136782824993134, 0.22653545439243317, 7.159047527238727e-05, 0.20698267221450806, 0.2584734559059143, 0.3793233633041382, 0.3888079822063446, 0.2165365368127823], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7999a2c1b668cfd6d047a2b2d720b2d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.039358366280794144, 0.3945406675338745, 0.0069579193368554115, 0.14127454161643982, 0.39049339294433594, 0.15763993561267853, 0.12085472792387009, 0.19900810718536377, 0.3689376413822174, 0.03470638394355774, 0.08821672201156616, 0.43156611919403076, 0.04575345665216446, 0.27736109495162964, 0.3500295877456665, 0.2050711214542389, 0.26294079422950745, 0.26341819763183594, 0.4499717652797699, 0.13273364305496216, 0.15146227180957794, 0.29820501804351807, 0.39637821912765503, 0.4589052200317383], dtype='float32').reshape([24]),
            paddle.to_tensor([0.31094861030578613, 0.43878790736198425, 0.3265332281589508, 0.03955431655049324, 0.3714279234409332, 0.2018395960330963, 0.27987831830978394, 0.12528540194034576, 0.042491935193538666, 0.4062608480453491, 0.2556559443473816, 0.46050363779067993, 0.3018836975097656, 0.14336811006069183, 0.4520810544490814, 0.24577213823795319, 0.05765274539589882, 0.0047959499061107635, 0.30322301387786865, 0.381804496049881, 0.46875905990600586, 0.4308786988258362, 0.24103906750679016, 0.1662113517522812], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5d42eb2dbdd2d2de5fc3ab34639f5bd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17592883110046387, 0.03469725325703621, 0.25857987999916077, 0.352712482213974, 0.36154285073280334, 0.05553416535258293, 0.1194208413362503, 0.24819424748420715, 0.4407810568809509, 0.13568349182605743, 0.3051833212375641, 0.04722002148628235, 0.12207715958356857, 0.07759536057710648, 0.47385936975479126, 0.4368453621864319, 0.017338952049613, 0.3319328725337982, 0.2324516624212265, 0.13104812800884247, 0.16603384912014008, 0.2207375168800354, 0.4661545753479004, 0.26476046442985535], dtype='float32').reshape([24]),
            paddle.to_tensor([0.25332576036453247, 0.13349002599716187, 0.3927764594554901, 0.4769271910190582, 0.3319157063961029, 0.3989616632461548, 0.28490033745765686, 0.44941404461860657, 0.3463718891143799, 0.03588806092739105, 0.08157992362976074, 0.4608616530895233, 0.41617417335510254, 0.15516746044158936, 0.38408249616622925, 0.005587894003838301, 0.2617878317832947, 0.2939292788505554, 0.3211876153945923, 0.09090942144393921, 0.10683024674654007, 0.3601209819316864, 0.3663609027862549, 0.4036162197589874], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e316f224e16a10c484e15e5bcdf07310(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.25599902868270874, 0.1396738439798355, 0.362801194190979, 0.39548158645629883, 0.25889021158218384, 0.2006731778383255, 0.4167245030403137, 0.4161047339439392, 0.1630173772573471, 0.3588005006313324, 0.273106187582016, 0.05884767696261406, 0.017136884853243828, 0.18808259069919586, 0.32606056332588196, 0.47483059763908386, 0.2074398696422577, 0.2892627418041229, 0.4575323462486267, 0.13849958777427673, 0.36717066168785095, 0.34571149945259094, 0.3694801926612854, 0.37464144825935364], dtype='float32').reshape([24]),
            paddle.to_tensor([0.3591902256011963, 0.016196629032492638, 0.15209847688674927, 0.3782707750797272, 0.08604823052883148, 0.10876551270484924, 0.3035818934440613, 0.34033504128456116, 0.3719601631164551, 0.18991534411907196, 0.2965843379497528, 0.20926766097545624, 0.1343764364719391, 0.3494398891925812, 0.06425300240516663, 0.18336115777492523, 0.08322586864233017, 0.38804125785827637, 0.23023860156536102, 0.3294859528541565, 0.2018568068742752, 0.24597583711147308, 0.11924022436141968, 0.42048680782318115], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eb582097850e4ac003bf193942c0ba63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.40099185705184937, 0.2456011176109314, 0.12352950870990753, 0.25216856598854065, 0.2630184292793274, 0.3151211440563202, 0.4328896105289459, 0.05237346887588501, 0.38539284467697144, 0.4673290252685547, 0.09777123481035233, 0.38350048661231995, 0.09805149585008621, 0.006997129879891872, 0.3231181502342224, 0.46741804480552673, 0.4412451386451721, 0.3880082666873932, 0.005674261599779129, 0.39295679330825806, 0.18637141585350037, 0.14476168155670166, 0.14057494699954987, 0.4187673032283783], dtype='float32').reshape([24]),
            paddle.to_tensor([0.4721943736076355, 0.22462841868400574, 0.45462802052497864, 0.3265056908130646, 0.44447705149650574, 0.4873240888118744, 0.3870282769203186, 0.2445104867219925, 0.20599457621574402, 0.29402902722358704, 0.3615923523902893, 0.1600436121225357, 0.12836667895317078, 0.23290058970451355, 0.3609430193901062, 0.049826480448246, 0.2798941433429718, 0.4837586283683777, 0.3053629994392395, 0.24311698973178864, 0.21264836192131042, 0.3632981479167938, 0.2687608301639557, 0.23331208527088165], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f45a6f4e0b54ba7010c01438a4c59a16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.15916624665260315, 0.30129116773605347, 0.4341990649700165, 0.08781782537698746, 0.35491085052490234, 0.3393780589103699, 0.04864159971475601, 0.1711697280406952, 0.18672475218772888, 0.3324241638183594, 0.17518575489521027, 0.26572081446647644, 0.3868301510810852, 0.18951106071472168, 0.1197492927312851, 0.06457968801259995, 0.38196906447410583, 0.08795572817325592, 0.12297769635915756, 0.439588338136673, 0.2740953862667084, 0.29074352979660034, 0.11280107498168945, 0.3202146291732788], dtype='float32').reshape([24]),
            paddle.to_tensor([0.03591357171535492, 0.3915875554084778, 0.3049783706665039, 0.45237651467323303, 0.2766997516155243, 0.07701076567173004, 0.101206474006176, 0.2556135952472687, 0.2429269403219223, 0.38586580753326416, 0.3399960994720459, 0.2913559079170227, 0.3584853708744049, 0.237050861120224, 0.46640413999557495, 0.209255188703537, 0.038706108927726746, 0.05576273053884506, 0.11080672591924667, 0.343860000371933, 0.3634796440601349, 0.48467907309532166, 0.023275084793567657, 0.03292467072606087], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bcdf89682f9a805e789a4e4c98478ad6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.038909077644348145, 0.3399308919906616, 0.3942354917526245, 0.024993911385536194, 0.23892943561077118, 0.4856584072113037, 0.12973549962043762, 0.11979029327630997, 0.10715901851654053, 0.09510361403226852, 0.025557542219758034, 0.34125566482543945, 0.3229012191295624, 0.09680626541376114, 0.05019602179527283, 0.08694581687450409, 0.27681317925453186, 0.2181738317012787, 0.10523960739374161, 0.3929179608821869, 0.19292506575584412, 0.1268174648284912, 0.10058706998825073, 0.039422422647476196], dtype='float32').reshape([24]),
            paddle.to_tensor([0.04009241238236427, 0.4719999134540558, 0.05338800325989723, 0.2771954834461212, 0.22848385572433472, 0.3336256742477417, 0.23608368635177612, 0.097867950797081, 0.3537161648273468, 0.04821527749300003, 0.48825299739837646, 0.08220426738262177, 0.3223990797996521, 0.1440141201019287, 0.2643798887729645, 0.08770913630723953, 0.11175554990768433, 0.3620840013027191, 0.40019500255584717, 0.3350338339805603, 0.37404751777648926, 0.3628470301628113, 0.14210949838161469, 0.05353432521224022], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_496f43bccb4de34d317c796d2386d090(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1574077159166336, 0.3334123492240906, 0.09426672011613846, 0.06949065625667572, 0.4422285556793213, 0.315021276473999, 0.1329808533191681, 0.383741170167923, 0.4856407046318054, 0.4788067936897278, 0.10565251111984253, 0.4423671364784241, 0.05679147690534592, 0.2548207640647888, 0.13908345997333527, 0.06867428869009018, 0.4008411169052124, 0.05972125753760338, 0.03979163244366646, 0.16385677456855774, 0.024615496397018433, 0.15653987228870392, 0.1204904094338417, 0.3101099729537964], dtype='float32').reshape([24]),
            paddle.to_tensor([0.18327464163303375, 0.2106115221977234, 0.4599856436252594, 0.2389131784439087, 0.39463287591934204, 0.2049688994884491, 0.35892537236213684, 0.11915910243988037, 0.06872140616178513, 0.24907539784908295, 0.377062052488327, 0.3357710838317871, 0.3412764072418213, 0.038652289658784866, 0.10039730370044708, 0.2673295736312866, 0.18875476717948914, 0.10459449142217636, 0.3070310652256012, 0.4416041374206543, 0.3604715168476105, 0.15391291677951813, 0.15516453981399536, 0.25252071022987366], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_08543f5a5c1ba01e8a00052a2ac57260(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.04878520593047142, 0.17960986495018005, 0.49700823426246643, 0.23891982436180115, 0.17502808570861816, 0.4426323175430298, 0.05405236780643463, 0.015351681038737297, 0.27742770314216614, 0.08839289844036102, 0.001497834688052535, 0.44509416818618774, 0.3125479817390442, 0.22210834920406342, 0.0676148384809494, 0.20860689878463745, 0.22676384449005127, 0.39132800698280334, 0.11458592116832733, 0.03387640789151192, 0.37026408314704895, 0.22858281433582306, 0.08239870518445969, 0.21815472841262817], dtype='float32').reshape([24]),
            paddle.to_tensor([0.06007709354162216, 0.053530313074588776, 0.28479912877082825, 0.11635381728410721, 0.2747719883918762, 0.07908884435892105, 0.423241525888443, 0.3920823633670807, 0.08847576379776001, 0.11656481772661209, 0.2965747117996216, 0.20334912836551666, 0.17061574757099152, 0.04140928015112877, 0.478849321603775, 0.2084011733531952, 0.16306337714195251, 0.08906558156013489, 0.179595485329628, 0.3444235622882843, 0.00749715231359005, 0.18442001938819885, 0.05504681169986725, 0.17166440188884735], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2491e2ec008cb05baed398ca5cdb41f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3719020187854767, 0.16279612481594086, 0.13391251862049103, 0.3414532542228699, 0.07134281098842621, 0.381379097700119, 0.21372012794017792, 0.12899459898471832, 0.018952375277876854, 0.24479302763938904, 0.4289688467979431, 0.45822930335998535, 0.2134883999824524, 0.11343982815742493, 0.13292600214481354, 0.09579435735940933, 0.41478431224823, 0.14802493155002594, 0.24160359799861908, 0.18166586756706238, 0.4009978771209717, 0.1671902984380722, 0.19765590131282806, 0.017324866726994514], dtype='float32').reshape([24]),
            paddle.to_tensor([0.2767189145088196, 0.485084593296051, 0.11646454781293869, 0.1308886706829071, 0.14616422355175018, 0.28169798851013184, 0.1651478409767151, 0.1612265259027481, 0.34223508834838867, 0.4733552634716034, 0.04274512827396393, 0.32972681522369385, 0.20572535693645477, 0.15287120640277863, 0.07068713009357452, 0.36129626631736755, 0.12649084627628326, 0.15651515126228333, 0.21436689794063568, 0.27790597081184387, 0.47011566162109375, 0.2592821717262268, 0.04462004080414772, 0.21350638568401337], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_592238c85d30361bf6bac43977d1c55a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2547762095928192, 0.3498023748397827, 0.3156878650188446, 0.4879373013973236, 0.2528179883956909, 0.32179269194602966, 0.38072264194488525, 0.2935139536857605, 0.04912817105650902, 0.395313024520874, 0.08545083552598953, 0.14934542775154114, 0.13591080904006958, 0.06065186485648155, 0.3273535370826721, 0.14702588319778442, 0.014861243776977062, 0.39115723967552185, 0.1579233705997467, 0.3674466907978058, 0.21116913855075836, 0.3814476430416107, 0.38902825117111206, 0.4511795938014984], dtype='float32').reshape([24]),
            paddle.to_tensor([0.1958245486021042, 0.18595807254314423, 0.46121615171432495, 0.21722836792469025, 0.4886452555656433, 0.3589939475059509, 0.28649839758872986, 0.4112842381000519, 0.17734384536743164, 0.11697845160961151, 0.3101177513599396, 0.41952264308929443, 0.12558411061763763, 0.4209752678871155, 0.4695117771625519, 0.28637826442718506, 0.09432416409254074, 0.07567030191421509, 0.21127890050411224, 0.10766322165727615, 0.20428743958473206, 0.4794478118419647, 0.45492061972618103, 0.3151381015777588], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fe0c4c1e1ce2851cc6a59a669e7a2e63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.48980507254600525, 0.20225262641906738, 0.04057382047176361, 0.33090081810951233, 0.2324766218662262, 0.15127529203891754, 0.3132180869579315, 0.45078277587890625, 0.09861063212156296, 0.16340070962905884, 0.31645265221595764, 0.4395062029361725, 0.42577478289604187, 0.24138735234737396, 0.41486769914627075, 0.16197338700294495, 0.4532359838485718, 0.16531799733638763, 0.4776124358177185, 0.2765220105648041, 0.2915506958961487, 0.010984274558722973, 0.344181627035141, 0.1449827253818512], dtype='float32').reshape([24]),
            paddle.to_tensor([0.2537020742893219, 0.4351765215396881, 0.40081095695495605, 0.29641634225845337, 0.08549311012029648, 0.4368189871311188, 0.33629897236824036, 0.03362615406513214, 0.47051942348480225, 0.34780511260032654, 0.2041577398777008, 0.3109087646007538, 0.04840646684169769, 0.3595603406429291, 0.09749352186918259, 0.11423201113939285, 0.46732190251350403, 0.42387592792510986, 0.4630972445011139, 0.1514451801776886, 0.3374304175376892, 0.32001638412475586, 0.4781065583229065, 0.28126949071884155], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_82e27c2656f6a525245f3a28ee3aee0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2204131782054901, 0.03807526081800461, 0.48926469683647156, 0.19799785315990448, 0.3907511234283447, 0.11003187298774719, 0.2395654171705246, 0.29793038964271545, 0.0872715637087822, 0.07126103341579437, 0.4106282591819763, 0.29929354786872864, 0.49663305282592773, 0.43654561042785645, 0.29005733132362366, 0.455079048871994, 0.3735303580760956, 0.0888025313615799, 0.33488884568214417, 0.27772805094718933, 0.15267568826675415, 0.1849013864994049, 0.06742595136165619, 0.4781872630119324], dtype='float32').reshape([24]),
            paddle.to_tensor([0.44875311851501465, 0.06154567748308182, 0.38379648327827454, 0.06762097775936127, 0.48407137393951416, 0.3723605275154114, 0.36683350801467896, 0.04954904690384865, 0.02068895660340786, 0.024976611137390137, 0.06447148323059082, 0.2950979173183441, 0.2849016785621643, 0.07215218991041183, 0.41507527232170105, 0.44508618116378784, 0.1184140145778656, 0.27914223074913025, 0.41455212235450745, 0.12697471678256989, 0.28466543555259705, 0.3009752035140991, 0.47303274273872375, 0.01803208328783512], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bf4a586aa0ed59d1fb7fc317a0127335(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2516383230686188, 0.047756925225257874, 0.03063337877392769, 0.11361321061849594, 0.24711111187934875, 0.08640994131565094, 0.2960216999053955, 0.22944676876068115, 0.3228716552257538, 0.3874753713607788, 0.21958103775978088, 0.13152103126049042, 0.007129548117518425, 0.445732980966568, 0.2601751387119293, 0.07645515352487564, 0.05519404262304306, 0.2225545048713684, 0.13739489018917084, 0.499168336391449, 0.04905478283762932, 0.4859500825405121, 0.3245089054107666, 0.06479968875646591], dtype='float32').reshape([24]),
            paddle.to_tensor([0.19572697579860687, 0.38306108117103577, 0.10550160706043243, 0.030066341161727905, 0.03104914352297783, 0.3735658824443817, 0.3557254374027252, 0.09984175860881805, 0.4012469947338104, 0.12817606329917908, 0.09454776346683502, 0.18820486962795258, 0.4995112419128418, 0.10901344567537308, 0.2803332805633545, 0.3449530601501465, 0.4299734830856323, 0.07380348443984985, 0.07215210795402527, 0.41033700108528137, 0.46160903573036194, 0.3456532955169678, 0.19057802855968475, 0.4349082410335541], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7a8da153bf46b9b8c6f1d3ec38d6e8d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.25554725527763367, 0.43168118596076965, 0.09809330850839615, 0.033004239201545715, 0.10878163576126099, 0.19198276102542877, 0.48066094517707825, 0.2239132970571518, 0.2479427456855774, 0.11032247543334961, 0.24195076525211334, 0.16153396666049957, 0.16218096017837524, 0.34178224205970764, 0.09137929975986481, 0.012891361489892006, 0.144174724817276, 0.4792724549770355, 0.3658965826034546, 0.21254576742649078, 0.27759334444999695, 0.02020539529621601, 0.3824334442615509, 0.08955428004264832], dtype='float32').reshape([24]),
            paddle.to_tensor([0.4210211932659149, 0.17378754913806915, 0.4390000104904175, 0.44076141715049744, 0.43140843510627747, 0.3801531493663788, 0.42686688899993896, 0.23885808885097504, 0.46993622183799744, 0.18452495336532593, 0.27816128730773926, 0.017498958855867386, 0.4972798526287079, 0.4798513352870941, 0.12299314141273499, 0.04091224819421768, 0.49109983444213867, 0.10709208250045776, 0.3041246831417084, 0.08253483474254608, 0.32801520824432373, 0.48760688304901123, 0.2133154571056366, 0.08507134765386581], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d1d1c94b128e11ace23283c48b09893a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3663907051086426, 0.35455647110939026, 0.2186216115951538, 0.08294322341680527, 0.06054316461086273, 0.1406097412109375, 0.11310704052448273, 0.16483837366104126, 0.04879584163427353, 0.11783308535814285, 0.13864853978157043, 0.29854729771614075, 0.31629136204719543, 0.41068923473358154, 0.3913551867008209, 0.29904869198799133, 0.3528215289115906, 0.44186416268348694, 0.4991440773010254, 0.28836682438850403, 0.2559347152709961, 0.1443294733762741, 0.007923737168312073, 0.39657965302467346], dtype='float32').reshape([24]),
            paddle.to_tensor([0.43868494033813477, 0.053570397198200226, 0.28789588809013367, 0.0493122860789299, 0.27466684579849243, 0.47885391116142273, 0.35287562012672424, 0.19078174233436584, 0.4644488990306854, 0.31196051836013794, 0.14379744231700897, 0.022491516545414925, 0.48410195112228394, 0.0745403915643692, 0.4248199462890625, 0.18228530883789062, 0.4969000816345215, 0.2985125780105591, 0.4252690076828003, 0.11471006274223328, 0.22579613327980042, 0.2814078629016876, 0.24145296216011047, 0.4465121626853943], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a573e00ea79fb85291e1ac2a7063ec6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4543927013874054, 0.2887659966945648, 0.2771182656288147, 0.16910862922668457, 0.22770319879055023, 0.22207997739315033, 0.3673876225948334, 0.03923211991786957, 0.12822340428829193, 0.002295473124831915, 0.13044649362564087, 0.3756214380264282, 0.3156859278678894, 0.44269824028015137, 0.30580219626426697, 0.15461517870426178, 0.24539262056350708, 0.026282576844096184, 0.26973599195480347, 0.29480624198913574, 0.1261141449213028, 0.14105457067489624, 0.3859430253505707, 0.4714984893798828], dtype='float32').reshape([24]),
            paddle.to_tensor([0.4200001657009125, 0.37605801224708557, 0.3129108250141144, 0.2710762321949005, 0.06828773766756058, 0.12413162738084793, 0.1646467000246048, 0.11383646726608276, 0.14136642217636108, 0.40018269419670105, 0.17496038973331451, 0.3614141047000885, 0.23492781817913055, 0.2573397159576416, 0.17333728075027466, 0.01121706422418356, 0.3146924674510956, 0.18423867225646973, 0.40561532974243164, 0.06707312911748886, 0.4913322925567627, 0.020940247923135757, 0.1494779884815216, 0.05654345825314522], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d64b9a51e4ce163cb49f106eedeeff46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.36723625659942627, 0.4219793975353241, 0.2395118921995163, 0.39058825373649597, 0.21673628687858582, 0.04407937079668045, 0.15127533674240112, 0.20488286018371582, 0.1764059215784073, 0.4978099763393402, 0.03785005211830139, 0.1496284008026123, 0.444339394569397, 0.3051510751247406, 0.32262101769447327, 0.2426222264766693, 0.17632147669792175, 0.12882472574710846, 0.48677319288253784, 0.39660948514938354, 0.06599558144807816, 0.001164985354989767, 0.41827720403671265, 0.06179322674870491], dtype='float32').reshape([24]),
            paddle.to_tensor([0.1887260377407074, 0.09598851203918457, 0.3264661729335785, 0.4065801203250885, 0.38987240195274353, 0.1903878003358841, 0.1631097048521042, 0.300360769033432, 0.3441500961780548, 0.06505870819091797, 0.33313700556755066, 0.04649403318762779, 0.4551711976528168, 0.1384315937757492, 0.3590712249279022, 0.11441738903522491, 0.37692493200302124, 0.2756788730621338, 0.23474092781543732, 0.3535337746143341, 0.29002466797828674, 0.4498613774776459, 0.18530374765396118, 0.01379961147904396], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_90d33e3e10797dec64fd0621a972f84e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2161395251750946, 0.3661425709724426, 0.46938997507095337, 0.0909985825419426, 0.36618661880493164, 0.40171000361442566, 0.2434929758310318, 0.21525532007217407, 0.011513223871588707, 0.035295430570840836, 0.3922966718673706, 0.34940701723098755, 0.03880881518125534, 0.08333639055490494, 0.14734575152397156, 0.3802799582481384, 0.058920178562402725, 0.4193512201309204, 0.2635982632637024, 0.4041859805583954, 0.46978452801704407, 0.284390389919281, 0.42282527685165405, 0.3870491683483124], dtype='float32').reshape([24]),
            paddle.to_tensor([0.03583795204758644, 0.16653122007846832, 0.004321552813053131, 0.34538742899894714, 0.0026336314622312784, 0.3622661828994751, 0.08812008053064346, 0.48882657289505005, 0.09329172223806381, 0.3659762740135193, 0.009717301465570927, 0.03383651748299599, 0.07134215533733368, 0.2950221002101898, 0.3951059877872467, 0.33725982904434204, 0.21132217347621918, 0.07368962466716766, 0.08601662516593933, 0.0923643484711647, 0.12226413935422897, 0.2507210373878479, 0.39674901962280273, 0.08643220365047455], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_008dbf44e49c61f0be1c082941fc6441(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2365260273218155, 0.45617425441741943, 0.4253959059715271, 0.2381434589624405, 0.12551376223564148, 0.4907849133014679, 0.24250657856464386, 0.12615105509757996, 0.040647100657224655, 0.12364145368337631, 0.17677100002765656, 0.34418466687202454, 0.37832024693489075, 0.01046819519251585, 0.15602649748325348, 0.16724339127540588, 0.49050188064575195, 0.46733924746513367, 0.19932927191257477, 0.029527762904763222, 0.30439087748527527, 0.27832484245300293, 0.29128706455230713, 0.21580353379249573], dtype='float32').reshape([24]),
            paddle.to_tensor([0.1150556430220604, 0.27425748109817505, 0.35669612884521484, 0.4079514145851135, 0.21220143139362335, 0.1703212410211563, 0.053285904228687286, 0.4234203100204468, 0.20476634800434113, 0.35055646300315857, 0.2255714386701584, 0.39724722504615784, 0.05438618361949921, 0.4757373034954071, 0.2652585506439209, 0.08283765614032745, 0.4978693127632141, 0.26089954376220703, 0.4675462543964386, 0.1152849942445755, 0.4780542552471161, 0.48625507950782776, 0.47919830679893494, 0.1496540755033493], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e1c4cb3730fab89759c17d607f0ca36d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.39540114998817444, 0.4052145183086395, 0.24481157958507538, 0.38127702474594116, 0.26841700077056885, 0.34546026587486267, 0.47020867466926575, 0.3479892611503601, 0.14807449281215668, 0.46927914023399353, 0.19036991894245148, 0.4771966338157654, 0.191911980509758, 0.36249521374702454, 0.1571909487247467, 0.1916063278913498, 0.06131294369697571, 0.155577152967453, 0.4286286234855652, 0.25168806314468384, 0.17827445268630981, 0.3215316832065582, 0.16960911452770233, 0.13326044380664825], dtype='float32').reshape([24]),
            paddle.to_tensor([0.012689746916294098, 0.4205149710178375, 0.4720805883407593, 0.1487337201833725, 0.32790321111679077, 0.4725497364997864, 0.4298741817474365, 0.4038618505001068, 0.1413702815771103, 0.3441394567489624, 0.29578638076782227, 0.15150883793830872, 0.12163267284631729, 0.12499600648880005, 0.2777296304702759, 0.265295147895813, 0.14899104833602905, 0.2085043489933014, 0.10016706585884094, 0.3347032070159912, 0.4612159729003906, 0.33212345838546753, 0.49622848629951477, 0.22084583342075348], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e7b46f82f847f5cf519851f7471fce9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.20074956119060516, 0.006004451774060726, 0.28077852725982666, 0.4546239376068115, 0.21417616307735443, 0.24597641825675964, 0.15045066177845, 0.06943879276514053, 0.15041644871234894, 0.4118548631668091, 0.317017525434494, 0.1997939497232437, 0.05939178913831711, 0.28186464309692383, 0.1327451765537262, 0.07069694995880127, 0.3132239580154419, 0.045049991458654404, 0.3539958596229553, 0.21250015497207642, 0.37754812836647034, 0.4742584228515625, 0.38393500447273254, 0.30397599935531616], dtype='float32').reshape([24]),
            paddle.to_tensor([0.2569807767868042, 0.3975311517715454, 0.3740795850753784, 0.3761407136917114, 0.4173624813556671, 0.26201045513153076, 0.2952626943588257, 0.43905067443847656, 0.3372174799442291, 0.3868354856967926, 0.20501548051834106, 0.0402902327477932, 0.47070422768592834, 0.37468600273132324, 0.3539433479309082, 0.08296366035938263, 0.49367520213127136, 0.4866485893726349, 0.23137977719306946, 0.40090855956077576, 0.3055494725704193, 0.3542640507221222, 0.10364445298910141, 0.4668697714805603], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_af3ef6210aac23d7a2f23ab212bbc68e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.24436210095882416, 0.1577892005443573, 0.32023486495018005, 0.4311057925224304, 0.2293730527162552, 0.14441902935504913, 0.19578328728675842, 0.350273460149765, 0.03376411274075508, 0.2338382601737976, 0.04354596138000488, 0.3282777667045593, 0.2418089359998703, 0.21038350462913513, 0.08402062207460403, 0.4877525269985199, 0.1305840164422989, 0.4571300745010376, 0.40948811173439026, 0.3138332962989807, 0.23291721940040588, 0.06096450239419937, 0.4724157750606537, 0.11062170565128326], dtype='float32').reshape([24]),
            paddle.to_tensor([0.08481891453266144, 0.39226698875427246, 0.20942898094654083, 0.4168529510498047, 0.134723961353302, 0.4427735507488251, 0.006096470169723034, 0.13637667894363403, 0.34508833289146423, 0.29255610704421997, 0.07818951457738876, 0.3503652811050415, 0.04966562241315842, 0.21193158626556396, 0.3886560797691345, 0.23403300344944, 0.08357207477092743, 0.22580283880233765, 0.32151517271995544, 0.46335485577583313, 0.36000823974609375, 0.33015701174736023, 0.46079301834106445, 0.42538461089134216], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4e7cba8ac2ef9f2cf2716a513ae79265(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4664541780948639, 0.007506831083446741, 0.38206496834754944, 0.017153361812233925, 0.017645400017499924, 0.464738667011261, 0.3660934865474701, 0.26948437094688416, 0.4750320017337799, 0.4335857331752777, 0.3793811500072479, 0.36705541610717773, 0.08372177183628082, 0.4025338292121887, 0.48793432116508484, 0.06428349763154984, 0.11960878223180771, 0.46028637886047363, 0.10398127138614655, 0.11168761551380157, 0.24067647755146027, 0.4723697304725647, 0.2210990935564041, 0.46436184644699097], dtype='float32').reshape([24]),
            paddle.to_tensor([0.21702764928340912, 0.22689451277256012, 0.17190660536289215, 0.11031419038772583, 0.08126579970121384, 0.32189399003982544, 0.2388981282711029, 0.34182074666023254, 0.43248140811920166, 0.42777159810066223, 0.014244680292904377, 0.00933239609003067, 0.2599926292896271, 0.29693177342414856, 0.46707698702812195, 0.0014438708312809467, 0.2947554588317871, 0.24876250326633453, 0.1448405534029007, 0.25375500321388245, 0.12856200337409973, 0.16168108582496643, 0.055015191435813904, 0.16082388162612915], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_081170bc4c505124cc52f3a877e78d09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.41944703459739685, 0.1386147290468216, 0.4366065561771393, 0.07455060631036758, 0.0120164779946208, 0.2896839678287506, 0.4581480920314789, 0.04055575653910637, 0.34591144323349, 0.326579749584198, 0.41891586780548096, 0.23125995695590973, 0.18032914400100708, 0.3510279059410095, 0.2586159110069275, 0.0994378849864006, 0.48573851585388184, 0.26900312304496765, 0.37603023648262024, 0.04380328953266144, 0.03939492627978325, 0.10328290611505508, 0.31529170274734497, 0.25040847063064575], dtype='float32').reshape([24]),
            paddle.to_tensor([0.24319596588611603, 0.45159703493118286, 0.03518538922071457, 0.47034338116645813, 0.39459022879600525, 0.2125380039215088, 0.34677252173423767, 0.28210514783859253, 0.3646690547466278, 0.15016362071037292, 0.01984117180109024, 0.4635527431964874, 0.3094014823436737, 0.4907504916191101, 0.1598355770111084, 0.03902144357562065, 0.41641244292259216, 0.17608290910720825, 0.1316220462322235, 0.0702786073088646, 0.42263975739479065, 0.4934868812561035, 0.17756617069244385, 0.34627804160118103], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_20e7e41c79341d6139f2ad02e85725cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3861091136932373, 0.38152921199798584, 0.44245272874832153, 0.35791584849357605, 0.3918468952178955, 0.061743754893541336, 0.1058834120631218, 0.4484610855579376, 0.15894366800785065, 0.13482604920864105, 0.11903544515371323, 0.07469668239355087, 0.01579664833843708, 0.05232267081737518, 0.0075988261960446835, 0.18294121325016022, 0.32161077857017517, 0.37411001324653625, 0.23486994206905365, 0.27800214290618896, 0.20532283186912537, 0.40364691615104675, 0.2356698364019394, 0.03704938292503357], dtype='float32').reshape([24]),
            paddle.to_tensor([0.19327662885189056, 0.2947341501712799, 0.44033724069595337, 0.07580437511205673, 0.023780709132552147, 0.2074434459209442, 0.42227211594581604, 0.32645970582962036, 0.19513602554798126, 0.3723924160003662, 0.16821089386940002, 0.4604695737361908, 0.2886887788772583, 0.269183486700058, 0.3298802375793457, 0.19749687612056732, 0.28431466221809387, 0.08667593449354172, 0.2036309689283371, 0.1638525426387787, 0.2755579650402069, 0.41632065176963806, 0.4455493986606598, 0.1434260904788971], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dec83aff657ea467652c5bce97f55951(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2928314507007599, 0.35736826062202454, 0.3843972086906433, 0.2649247348308563, 0.14423112571239471, 0.03819414600729942, 0.0928710401058197, 0.28775978088378906, 0.4608475863933563, 0.2693670690059662, 0.3294244408607483, 0.344162255525589, 0.37784212827682495, 0.16462473571300507, 0.06340683996677399, 0.42480865120887756, 0.2811238169670105, 0.45677798986434937, 0.37028244137763977, 0.29063403606414795, 0.2464531660079956, 0.3579019606113434, 0.0655788779258728, 0.35769593715667725], dtype='float32').reshape([24]),
            paddle.to_tensor([0.18442419171333313, 0.022861553356051445, 0.4484763443470001, 0.09234350174665451, 0.3241155743598938, 0.43141359090805054, 0.4819016754627228, 0.22271589934825897, 0.4861607253551483, 0.3105262815952301, 0.26027345657348633, 0.38955193758010864, 0.3826269507408142, 0.2974110245704651, 0.19895729422569275, 0.3816852569580078, 0.11880843341350555, 0.02665766142308712, 0.43944481015205383, 0.45559701323509216, 0.213868647813797, 0.4013108015060425, 0.060752253979444504, 0.342435747385025], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_67c6b0d6ccfbb1178ffb24f1702a6bab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.29382506012916565, 0.15121017396450043, 0.021513985469937325, 0.2047869861125946, 0.1828564703464508, 0.2566147446632385, 0.11570632457733154, 0.3820118010044098, 0.389024555683136, 0.18201781809329987, 0.4405120313167572, 0.4682217836380005, 0.051292434334754944, 0.13645558059215546, 0.49361974000930786, 0.18465663492679596, 0.2594228684902191, 0.22980239987373352, 0.18726837635040283, 0.4330078065395355, 0.22184620797634125, 0.44827374815940857, 0.46965956687927246, 0.07652278989553452], dtype='float32').reshape([24]),
            paddle.to_tensor([0.43856486678123474, 0.0906871110200882, 0.24827851355075836, 0.06989187747240067, 0.3102608919143677, 0.4121992588043213, 0.012084610760211945, 0.3237530589103699, 0.1812833696603775, 0.16134187579154968, 0.09543397277593613, 0.3766065537929535, 0.03647255524992943, 0.44298118352890015, 0.08183704316616058, 0.30951303243637085, 0.2864041030406952, 0.3509060740470886, 0.121591717004776, 0.4996446967124939, 0.171026349067688, 0.1457144021987915, 0.3155284821987152, 0.11363870650529861], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f809dbf98e6428147ab7a999a02784e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.395257830619812, 0.4812893271446228, 0.4035157561302185, 0.4596901535987854, 0.06588728725910187, 0.09194746613502502, 0.1580587774515152, 0.494392991065979, 0.3764691948890686, 0.3706943988800049, 0.46913158893585205, 0.21274416148662567, 0.3113678991794586, 0.49711745977401733, 0.377505898475647, 0.49972251057624817, 0.2703377604484558, 0.32867109775543213, 0.0361931212246418, 0.1302865892648697, 0.28577715158462524, 0.18079304695129395, 0.17841191589832306, 0.2649787962436676], dtype='float32').reshape([24]),
            paddle.to_tensor([0.18403877317905426, 0.06950866430997849, 0.029144462198019028, 0.08015970140695572, 0.28491443395614624, 0.34381425380706787, 0.12337134778499603, 0.27112841606140137, 0.22037489712238312, 0.14318911731243134, 0.2804096043109894, 0.058468129485845566, 0.4570331573486328, 0.49772459268569946, 0.4822782576084137, 0.096678227186203, 0.38325825333595276, 0.19170266389846802, 0.388616144657135, 0.32544660568237305, 0.19152146577835083, 0.017113853245973587, 0.016243204474449158, 0.4025624692440033], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_65ed5187da85d713b8e1535f54bc1ebe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.053806185722351074, 0.45891696214675903, 0.24312211573123932, 0.46730852127075195, 0.3346273899078369, 0.3496006429195404, 0.41284483671188354, 0.03566724807024002, 0.07910394668579102, 0.43081507086753845, 0.07702542841434479, 0.39681240916252136, 0.12910319864749908, 0.48178085684776306, 0.06116705760359764, 0.24829423427581787, 0.1831100583076477, 0.06923554837703705, 0.20056380331516266, 0.36568862199783325, 0.18942758440971375, 0.22129420936107635, 0.39158546924591064, 0.3310197591781616], dtype='float32').reshape([24]),
            paddle.to_tensor([0.2112686038017273, 0.03891182690858841, 0.2521170973777771, 0.4553987681865692, 0.4649070203304291, 0.003925838507711887, 0.10582482069730759, 0.17880779504776, 0.1761930286884308, 0.114957794547081, 0.25098395347595215, 0.2860599458217621, 0.12872907519340515, 0.4109175205230713, 0.32581230998039246, 0.18088020384311676, 0.45978832244873047, 0.20228692889213562, 0.29265594482421875, 0.020767753943800926, 0.3754936158657074, 0.24707840383052826, 0.2282084971666336, 0.12432954460382462], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eed23a69e9a0feab1d69a3aeb5a77c7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.33339065313339233, 0.44183942675590515, 0.1663869470357895, 0.06813167035579681, 0.005714530125260353, 0.25634947419166565, 0.06586265563964844, 0.34697291254997253, 0.2876550853252411, 0.493533194065094, 0.4655020534992218, 0.1096431091427803, 0.3993321359157562, 0.025697045028209686, 0.2377190738916397, 0.134072408080101, 0.4011228084564209, 0.23941615223884583, 0.16872653365135193, 0.1597764790058136, 0.012867043726146221, 0.4758508503437042, 0.1426258236169815, 0.49743106961250305], dtype='float32').reshape([24]),
            paddle.to_tensor([0.24940602481365204, 0.13710568845272064, 0.19387346506118774, 0.22800549864768982, 0.4168940782546997, 0.05018024519085884, 0.14784198999404907, 0.4305501878261566, 0.1470329314470291, 0.16453875601291656, 0.1588393747806549, 0.17451587319374084, 0.056585341691970825, 0.14240014553070068, 0.26411300897598267, 0.35325172543525696, 0.43812692165374756, 0.28621286153793335, 0.07554935663938522, 0.26163265109062195, 0.46818089485168457, 0.2802594006061554, 0.31799694895744324, 0.0047715893015265465], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_683204789f72de73303f857841995cd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.46141359210014343, 0.03614059090614319, 0.12426874786615372, 0.2589278519153595, 0.20482587814331055, 0.31854701042175293, 0.44225865602493286, 0.016221484169363976, 0.050999388098716736, 0.3883713483810425, 0.2552471458911896, 0.31796208024024963, 0.40085723996162415, 0.10466177761554718, 0.401383638381958, 0.34579214453697205, 0.21887338161468506, 0.4509848952293396, 0.006501639261841774, 0.11555369943380356, 0.2425161749124527, 0.2873491942882538, 0.2722403109073639, 0.22372837364673615], dtype='float32').reshape([24]),
            paddle.to_tensor([0.26650741696357727, 0.45501917600631714, 0.2576369643211365, 0.4583207368850708, 0.4981445074081421, 0.00787600502371788, 0.14057868719100952, 0.42438650131225586, 0.10331928730010986, 0.40191650390625, 0.3945758044719696, 0.4829796254634857, 0.0986485555768013, 0.47333866357803345, 0.04450336843729019, 0.006029950920492411, 0.14278480410575867, 0.36358505487442017, 0.03841891512274742, 0.2598845064640045, 0.07067080587148666, 0.3645738959312439, 0.2139510065317154, 0.3837554156780243], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_45d4c046503efd6fffc49301e37346e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.18328790366649628, 0.13790543377399445, 0.12096403539180756, 0.20254115760326385, 0.42318788170814514, 0.22869712114334106, 0.41029515862464905, 0.31199973821640015, 0.00413438631221652, 0.47484201192855835, 0.23830583691596985, 0.27385443449020386, 0.3253830671310425, 0.41182297468185425, 0.49252840876579285, 0.23393718898296356, 0.0982903316617012, 0.04956307262182236, 0.00015125394565984607, 0.3625272512435913, 0.09275129437446594, 0.04547438025474548, 0.14088954031467438, 0.2993544936180115], dtype='float32').reshape([24]),
            paddle.to_tensor([0.3733654320240021, 0.4288122057914734, 0.49884992837905884, 0.1379363238811493, 0.25350549817085266, 0.033080145716667175, 0.4714765250682831, 0.27251046895980835, 0.12945646047592163, 0.2188611924648285, 0.03092116490006447, 0.11386024951934814, 0.12772217392921448, 0.2810160517692566, 0.07152555882930756, 0.22567634284496307, 0.20455937087535858, 0.2413434386253357, 0.17944936454296112, 0.3796754777431488, 0.10451704263687134, 0.17536409199237823, 0.2761671245098114, 0.20975884795188904], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ae8f35ff730301d4f2dc010128a039c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.43386515974998474, 0.41544175148010254, 0.23280422389507294, 0.2798287868499756, 0.4644755423069, 0.31365254521369934, 0.3029618561267853, 0.05555473640561104, 0.4828035235404968, 0.42244642972946167, 0.45691871643066406, 0.41013455390930176, 0.004918091464787722, 0.30703288316726685, 0.25551971793174744, 0.34249669313430786, 0.2859838008880615, 0.37832531332969666, 0.21679389476776123, 0.3733070194721222, 0.2812666893005371, 0.31351467967033386, 0.033778008073568344, 0.3509100079536438], dtype='float32').reshape([24]),
            paddle.to_tensor([0.29477599263191223, 0.178890198469162, 0.20766042172908783, 0.4733906686306, 0.2941227853298187, 0.17971163988113403, 0.4343111515045166, 0.30361542105674744, 0.46589261293411255, 0.4885789155960083, 0.1085309162735939, 0.3114018142223358, 0.2864786684513092, 0.17709586024284363, 0.20329028367996216, 0.28622037172317505, 0.4018328785896301, 0.3139972984790802, 0.33059030771255493, 0.13680186867713928, 0.2944427728652954, 0.19074587523937225, 0.38746678829193115, 0.1618432104587555], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_67cde66cafd8cc57a7493537954bdcd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4312141239643097, 0.13264116644859314, 0.1605023890733719, 0.040225494652986526, 0.4392125606536865, 0.17857183516025543, 0.4638766944408417, 0.3806072771549225, 0.07603641599416733, 0.02092881128191948, 0.2647075951099396, 0.1716555505990982, 0.15668885409832, 0.29378581047058105, 0.4653826653957367, 0.39949214458465576, 0.19621102511882782, 0.4657381772994995, 0.00622948445379734, 0.048310816287994385, 0.018053891137242317, 0.04316616430878639, 0.3190076947212219, 0.10114273428916931], dtype='float32').reshape([24]),
            paddle.to_tensor([0.052838150411844254, 0.1513373702764511, 0.40824422240257263, 0.46992865204811096, 0.3140195608139038, 0.4865038990974426, 0.42771637439727783, 0.19449453055858612, 0.1553611159324646, 0.17337357997894287, 0.18066655099391937, 0.1265137940645218, 0.15326054394245148, 0.4968477487564087, 0.453971266746521, 0.4524460732936859, 0.1557072401046753, 0.46405014395713806, 0.07133635133504868, 0.08057336509227753, 0.04825890064239502, 0.43040141463279724, 0.31876319646835327, 0.08831211179494858], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_db6fbccd9a2dd602031503e25ab7b59a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4691127836704254, 0.21587099134922028, 0.17382203042507172, 0.3377580940723419, 0.23200176656246185, 0.44011011719703674, 0.312603235244751, 0.44923728704452515, 0.007605915889143944, 0.22613970935344696, 0.26542726159095764, 0.19376201927661896, 0.13222761452198029, 0.38347360491752625, 0.00841373298317194, 0.09500807523727417, 0.47308796644210815, 0.3806673288345337, 0.3817708492279053, 0.2833719849586487, 0.1323445588350296, 0.48738670349121094, 0.404735803604126, 0.014653407037258148], dtype='float32').reshape([24]),
            paddle.to_tensor([0.006123330444097519, 0.3217849135398865, 0.19295702874660492, 0.0912381187081337, 0.3869273066520691, 0.026838503777980804, 0.41028082370758057, 0.09869623184204102, 0.03278975561261177, 0.06976163387298584, 0.013199491426348686, 0.2985546588897705, 0.12086675316095352, 0.28622788190841675, 0.3507384657859802, 0.25793763995170593, 0.24174971878528595, 0.32617971301078796, 0.10887044668197632, 0.3099299371242523, 0.14295899868011475, 0.16366004943847656, 0.011801715940237045, 0.23189924657344818], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_af7389f72c828830d97e2c900fa653e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.27697497606277466, 0.345083624124527, 0.10970999300479889, 0.15074454247951508, 0.16114360094070435, 0.4411676526069641, 0.04893328994512558, 0.07032536715269089, 0.12416011840105057, 0.09644624590873718, 0.17378029227256775, 0.4338844418525696, 0.14427806437015533, 0.19335106015205383, 0.2758598029613495, 0.49349841475486755, 0.426896870136261, 0.21997860074043274, 0.40885117650032043, 0.3530758023262024, 0.14571242034435272, 0.16428513824939728, 0.2193647176027298, 0.4398437440395355], dtype='float32').reshape([24]),
            paddle.to_tensor([0.3287707269191742, 0.2933168113231659, 0.031689539551734924, 0.476423054933548, 0.4194153845310211, 0.330296128988266, 0.03839683532714844, 0.291202574968338, 0.21126292645931244, 0.2018045037984848, 0.03757752105593681, 0.44595077633857727, 0.4742007553577423, 0.3286994993686676, 0.4370025098323822, 0.10208725184202194, 0.06700240820646286, 0.38773590326309204, 0.025147220119833946, 0.13888688385486603, 0.2952336370944977, 0.39112916588783264, 0.11948025971651077, 0.2812141478061676], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a9dbd04f861666fd61385ea2cff09ffc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4314621388912201, 0.2626471519470215, 0.22205787897109985, 0.15510892868041992, 0.4067836105823517, 0.07495203614234924, 0.10712786018848419, 0.1160012036561966, 0.24771995842456818, 0.006325103342533112, 0.12690584361553192, 0.4916941225528717, 0.34031912684440613, 0.04771795868873596, 0.38993507623672485, 0.36269858479499817, 0.13252438604831696, 0.18508994579315186, 0.03712392598390579, 0.2665572762489319, 0.4886528551578522, 0.15469200909137726, 0.3323485851287842, 0.2218673825263977], dtype='float32').reshape([24]),
            paddle.to_tensor([0.29170775413513184, 0.17818447947502136, 0.2120242565870285, 0.07960209995508194, 0.40622198581695557, 0.03617842495441437, 0.24670298397541046, 0.05763398855924606, 0.2749381959438324, 0.2681066691875458, 0.4478433430194855, 0.27427318692207336, 0.4001559317111969, 0.21668653190135956, 0.1972852200269699, 0.1040903702378273, 0.4223109483718872, 0.47960230708122253, 0.42504045367240906, 0.09620174020528793, 0.4985962212085724, 0.3962371349334717, 0.4192229211330414, 0.3840426504611969], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d665061169523629b16a34b32f4ab09b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.45788344740867615, 0.00045639550080522895, 0.39150288701057434, 0.3667545020580292, 0.14908815920352936, 0.3052982687950134, 0.17156554758548737, 0.2143753319978714, 0.3832223415374756, 0.3176027536392212, 0.28317880630493164, 0.1692972034215927, 0.3962440490722656, 0.030912647023797035, 0.4081241190433502, 0.31854891777038574, 0.34653472900390625, 0.21547172963619232, 0.27609989047050476, 0.22581534087657928, 0.1000174731016159, 0.057527799159288406, 0.4991688132286072, 0.21601103246212006], dtype='float32').reshape([24]),
            paddle.to_tensor([0.06386272609233856, 0.08633637428283691, 0.4770280420780182, 0.09042476117610931, 0.3519243896007538, 0.3738279640674591, 0.24946142733097076, 0.2717282772064209, 0.0031764155719429255, 0.22161734104156494, 0.4693774878978729, 0.31049254536628723, 0.4913512170314789, 0.10404103994369507, 0.4772428274154663, 0.4393569529056549, 0.13143523037433624, 0.29769647121429443, 0.028460592031478882, 0.06796008348464966, 0.33599817752838135, 0.43803369998931885, 0.35787320137023926, 0.34984827041625977], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
class TestPrimitiveOp_5cad56c365cc8b4ad959ec3e6163a443(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8849179f5d074ef8df50d7e13dc2683f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 512], dtype='float16', min=0, max=0.5),
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
class TestPrimitiveOp_5878d475c12ff158016d91abe16b825d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.12419048696756363, 0.37535029649734497, 0.2765030562877655, 0.3821343183517456, 0.33024221658706665, 0.07229235023260117, 0.45719510316848755, 0.3944741487503052, 0.20532748103141785, 0.05674845352768898, 0.12264791131019592, 0.3296707570552826, 0.40370532870292664, 0.21281811594963074, 0.3262869417667389, 0.01108761690557003, 0.4206697940826416, 0.46054691076278687, 0.27791014313697815, 0.1937817484140396, 0.19722622632980347, 0.08373107016086578, 0.16766349971294403, 0.2599410116672516], dtype='float32').reshape([24]),
            paddle.to_tensor([0.47431719303131104, 0.13286490738391876, 0.3111533522605896, 0.36972784996032715, 0.4015478491783142, 0.36506417393684387, 0.38756969571113586, 0.1611185073852539, 0.1466299444437027, 0.22819964587688446, 0.2328542172908783, 0.3959265947341919, 0.05855312570929527, 0.2452484518289566, 0.28116732835769653, 0.03424753248691559, 0.23618049919605255, 0.3465275168418884, 0.3895937204360962, 0.1697375327348709, 0.3248807489871979, 0.4428333640098572, 0.28583231568336487, 0.08076059073209763], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
class TestPrimitiveOp_e652c96361b8b37785ec3f5092914967(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.029846541583538055, 0.4518411159515381, 0.1716068685054779, 0.2362489104270935, 0.3543258011341095, 0.40037932991981506, 0.23541198670864105, 0.167933389544487, 0.03557004779577255, 0.19448260962963104, 0.41416239738464355, 0.3681526482105255, 0.24877628684043884, 0.36061155796051025, 0.3229597210884094, 0.061476729810237885, 0.13109542429447174, 0.12173473089933395, 0.47740957140922546, 0.1820772886276245, 0.3760887086391449, 0.361057847738266, 0.2961604595184326, 0.34885305166244507], dtype='float32').reshape([24]),
            paddle.to_tensor([0.19965840876102448, 0.1485452502965927, 0.39907726645469666, 0.4856626093387604, 0.20386353135108948, 0.47867971658706665, 0.4497002363204956, 0.23218807578086853, 0.21636004745960236, 0.21615320444107056, 0.03118315525352955, 0.1251886934041977, 0.4594854712486267, 0.09600412845611572, 0.12832793593406677, 0.2248455286026001, 0.4104636013507843, 0.42970868945121765, 0.13298679888248444, 0.10439467430114746, 0.232858344912529, 0.40402474999427795, 0.4911373257637024, 0.23328770697116852], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5071efee0a6be317305f2d0225a47dd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.06122194975614548, 0.02137068659067154, 0.14039559662342072, 0.04391301050782204, 0.31955987215042114, 0.13062553107738495, 0.4961921274662018, 0.14633387327194214, 0.48925697803497314, 0.263266384601593, 0.261473685503006, 0.25453272461891174, 0.20250068604946136, 0.3981817960739136, 0.24953901767730713, 0.42823174595832825, 0.13289561867713928, 0.09234563261270523, 0.49178290367126465, 0.11917651444673538, 0.4463011920452118, 0.11164161562919617, 0.1418273001909256, 0.40878525376319885], dtype='float32').reshape([24]),
            paddle.to_tensor([0.47853320837020874, 0.471259206533432, 0.483436644077301, 0.15336103737354279, 0.22569091618061066, 0.3043261468410492, 0.43447503447532654, 0.2452462911605835, 0.49709680676460266, 0.4487230181694031, 0.425223708152771, 0.3794567883014679, 0.3276847004890442, 0.28613367676734924, 0.4307856261730194, 0.2105758637189865, 0.2616255581378937, 0.4875328540802002, 0.17232069373130798, 0.1616373360157013, 0.29326510429382324, 0.35090410709381104, 0.3807741105556488, 0.015944497659802437], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4795547951f4a225a51e3bae79aefb2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4213341772556305, 0.3540722131729126, 0.05336308851838112, 0.47785329818725586, 0.29147499799728394, 0.3708406090736389, 0.4891560673713684, 0.329334020614624, 0.44010481238365173, 0.01798633672297001, 0.02813560888171196, 0.15297915041446686, 0.4731057286262512, 0.02022893913090229, 0.43038150668144226, 0.05641293525695801, 0.05831751227378845, 0.46261438727378845, 0.35460519790649414, 0.170273557305336, 0.15965817868709564, 0.11220718920230865, 0.339863657951355, 0.26430243253707886], dtype='float32').reshape([24]),
            paddle.to_tensor([0.24000950157642365, 0.2011338770389557, 0.3849775791168213, 0.1800290048122406, 0.050875041633844376, 0.04831603541970253, 0.37717726826667786, 0.1439453512430191, 0.025947287678718567, 0.44146913290023804, 0.3924444317817688, 0.06777292490005493, 0.11720570176839828, 0.3823343515396118, 0.34130412340164185, 0.4829157888889313, 0.08625774085521698, 0.04810171201825142, 0.0038659004494547844, 0.2229214608669281, 0.04871803894639015, 0.06826524436473846, 0.00794372707605362, 0.05092710256576538], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c19bcbe6e4585407f2f92b634c3daae1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2705211341381073, 0.315491259098053, 0.02433658577501774, 0.3883836269378662, 0.1275789588689804, 0.1928188055753708, 0.37481293082237244, 0.2850351631641388, 0.060913003981113434, 0.11846338957548141, 0.05339489504694939, 0.32055607438087463, 0.3577638864517212, 0.19332900643348694, 0.3850342929363251, 0.3389863669872284, 0.4675685167312622, 0.1080484688282013, 0.38795146346092224, 0.3727991282939911, 0.46291935443878174, 0.49592286348342896, 0.16096989810466766, 0.2642364799976349], dtype='float32').reshape([24]),
            paddle.to_tensor([0.26355230808258057, 0.21223752200603485, 0.3469664752483368, 0.35968858003616333, 0.18014700710773468, 0.46183931827545166, 0.0240391343832016, 0.03798052296042442, 0.19690252840518951, 0.23696187138557434, 0.46961963176727295, 0.3380269408226013, 0.13394588232040405, 0.434874564409256, 0.14233030378818512, 0.0330117866396904, 0.31378623843193054, 0.42210060358047485, 0.018485013395547867, 0.4958232343196869, 0.12212290614843369, 0.4715007245540619, 0.21444234251976013, 0.37514764070510864], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d3739231ed19643d658495d0ac3bfd76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.08258425444364548, 0.0734853520989418, 0.09411701560020447, 0.05579982325434685, 0.06133255735039711, 0.39629992842674255, 0.4806704521179199, 0.4823877215385437, 0.12186381220817566, 0.4359857738018036, 0.23751281201839447, 0.30263516306877136, 0.30346933007240295, 0.2507888972759247, 0.03573431447148323, 0.27259814739227295, 0.19633232057094574, 0.1154838502407074, 0.26075252890586853, 0.16868750751018524, 0.053940530866384506, 0.29893261194229126, 0.19299538433551788, 0.17447715997695923], dtype='float32').reshape([24]),
            paddle.to_tensor([0.4466041922569275, 0.17117729783058167, 0.0720098540186882, 0.40316107869148254, 0.10955902189016342, 0.03818371146917343, 0.03343240171670914, 0.2106536626815796, 0.07636027038097382, 0.36251890659332275, 0.03222961351275444, 0.12276952713727951, 0.21909861266613007, 0.2954011857509613, 0.15671183168888092, 0.23567405343055725, 0.4272143542766571, 0.4417370855808258, 0.03791682422161102, 0.15880565345287323, 0.21285854279994965, 0.48467591404914856, 0.4582032561302185, 0.44239532947540283], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_de66b38941efb792b8a801fd74bfac0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.02179430052638054, 0.35659918189048767, 0.10288295149803162, 0.367136150598526, 0.03861471638083458, 0.3303942382335663, 0.02263519912958145, 0.08006787300109863, 0.39125028252601624, 0.07442966848611832, 0.4011555016040802, 0.3440481126308441, 0.4099203050136566, 0.08449126034975052, 0.2607165575027466, 0.46883636713027954, 0.010827611200511456, 0.47674131393432617, 0.01961446925997734, 0.23594188690185547, 0.4052788019180298, 0.062286511063575745, 0.12254875153303146, 0.15587499737739563], dtype='float32').reshape([24]),
            paddle.to_tensor([0.048090286552906036, 0.04565330594778061, 0.2716093063354492, 0.23335745930671692, 0.007170975673943758, 0.20589829981327057, 0.1202099472284317, 0.09331848472356796, 0.22194841504096985, 0.4402906894683838, 0.21088039875030518, 0.2652601897716522, 0.37642964720726013, 0.2923552989959717, 0.4936602711677551, 0.1778169572353363, 0.1487603783607483, 0.3199818730354309, 0.2537340223789215, 0.18421567976474762, 0.33916833996772766, 0.2629408538341522, 0.23394381999969482, 0.13371197879314423], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_68c9a95b3c2610ad2f3bc2a166b0cf24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4390544295310974, 0.19970671832561493, 0.2590944766998291, 0.07582758367061615, 0.34973135590553284, 0.14893846213817596, 0.1972925364971161, 0.10873378068208694, 0.4033542275428772, 0.4635089039802551, 0.039704035967588425, 0.2580186724662781, 0.30740436911582947, 0.07510010153055191, 0.43295150995254517, 0.2231997400522232, 0.2723812460899353, 0.467327356338501, 0.37129393219947815, 0.21167562901973724, 0.37400370836257935, 0.4357066750526428, 0.4713532030582428, 0.2917383909225464], dtype='float32').reshape([24]),
            paddle.to_tensor([0.1983165144920349, 0.3643958270549774, 0.26831236481666565, 0.2816506028175354, 0.1044212281703949, 0.17534904181957245, 0.45651739835739136, 0.20121921598911285, 0.11607038974761963, 0.4561237394809723, 0.23605439066886902, 0.12967082858085632, 0.295977383852005, 0.37622329592704773, 0.21720823645591736, 0.2169981151819229, 0.1342230588197708, 0.21265734732151031, 0.4741648733615875, 0.22956861555576324, 0.14797234535217285, 0.36484959721565247, 0.31306618452072144, 0.08679403364658356], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7f0546913417f973d0fc21f6bd192028(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.28779855370521545, 0.29507631063461304, 0.08921849727630615, 0.4362853765487671, 0.1639026254415512, 0.12181118130683899, 0.0925721600651741, 0.3030030429363251, 0.4857131540775299, 0.016851555556058884, 0.4665258228778839, 0.1737879514694214, 0.43905505537986755, 0.36795371770858765, 0.27491095662117004, 0.4894116222858429, 0.27295732498168945, 0.295064240694046, 0.20313318073749542, 0.05314280837774277, 0.07289009541273117, 0.37172722816467285, 0.20052076876163483, 0.2198108732700348], dtype='float32').reshape([24]),
            paddle.to_tensor([0.19671781361103058, 0.10109605640172958, 0.04864858463406563, 0.09422617405653, 0.3193044662475586, 0.4781797230243683, 0.28611403703689575, 0.25401005148887634, 0.10111551731824875, 0.017606079578399658, 0.04949459433555603, 0.16216453909873962, 0.15200424194335938, 0.24645820260047913, 0.21513457596302032, 0.10045173019170761, 0.3924245834350586, 0.37487998604774475, 0.03570882976055145, 0.015446366742253304, 0.033180106431245804, 0.2750890851020813, 0.17604152858257294, 0.2768828272819519], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8f934907cdf3828e0fc008311b7d6b4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4905850291252136, 0.1417711228132248, 0.3972661793231964, 0.26160258054733276, 0.2209509015083313, 0.040283795446157455, 0.2229292243719101, 0.365222692489624, 0.0916929617524147, 0.36036568880081177, 0.30587178468704224, 0.31058618426322937, 0.10337479412555695, 0.29634565114974976, 0.0766269713640213, 0.01825917512178421, 0.02897103875875473, 0.31862297654151917, 0.320491760969162, 0.23870548605918884, 0.1693069189786911, 0.004184379708021879, 0.29679298400878906, 0.03416705131530762], dtype='float32').reshape([24]),
            paddle.to_tensor([0.35126546025276184, 0.1637580841779709, 0.11904033273458481, 0.22047705948352814, 0.18035081028938293, 0.29125672578811646, 0.15367192029953003, 0.29683148860931396, 0.3086322546005249, 0.2522909939289093, 0.2270323932170868, 0.13263821601867676, 0.026715775951743126, 0.2140287458896637, 0.19878646731376648, 0.35739991068840027, 0.15136782824993134, 0.22653545439243317, 7.159047527238727e-05, 0.20698267221450806, 0.2584734559059143, 0.3793233633041382, 0.3888079822063446, 0.2165365368127823], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5724161870bf0c58d09b61a1a48aa2a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.039358366280794144, 0.3945406675338745, 0.0069579193368554115, 0.14127454161643982, 0.39049339294433594, 0.15763993561267853, 0.12085472792387009, 0.19900810718536377, 0.3689376413822174, 0.03470638394355774, 0.08821672201156616, 0.43156611919403076, 0.04575345665216446, 0.27736109495162964, 0.3500295877456665, 0.2050711214542389, 0.26294079422950745, 0.26341819763183594, 0.4499717652797699, 0.13273364305496216, 0.15146227180957794, 0.29820501804351807, 0.39637821912765503, 0.4589052200317383], dtype='float32').reshape([24]),
            paddle.to_tensor([0.31094861030578613, 0.43878790736198425, 0.3265332281589508, 0.03955431655049324, 0.3714279234409332, 0.2018395960330963, 0.27987831830978394, 0.12528540194034576, 0.042491935193538666, 0.4062608480453491, 0.2556559443473816, 0.46050363779067993, 0.3018836975097656, 0.14336811006069183, 0.4520810544490814, 0.24577213823795319, 0.05765274539589882, 0.0047959499061107635, 0.30322301387786865, 0.381804496049881, 0.46875905990600586, 0.4308786988258362, 0.24103906750679016, 0.1662113517522812], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f68dc7b7010f858a2420cf42fd279e8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17592883110046387, 0.03469725325703621, 0.25857987999916077, 0.352712482213974, 0.36154285073280334, 0.05553416535258293, 0.1194208413362503, 0.24819424748420715, 0.4407810568809509, 0.13568349182605743, 0.3051833212375641, 0.04722002148628235, 0.12207715958356857, 0.07759536057710648, 0.47385936975479126, 0.4368453621864319, 0.017338952049613, 0.3319328725337982, 0.2324516624212265, 0.13104812800884247, 0.16603384912014008, 0.2207375168800354, 0.4661545753479004, 0.26476046442985535], dtype='float32').reshape([24]),
            paddle.to_tensor([0.25332576036453247, 0.13349002599716187, 0.3927764594554901, 0.4769271910190582, 0.3319157063961029, 0.3989616632461548, 0.28490033745765686, 0.44941404461860657, 0.3463718891143799, 0.03588806092739105, 0.08157992362976074, 0.4608616530895233, 0.41617417335510254, 0.15516746044158936, 0.38408249616622925, 0.005587894003838301, 0.2617878317832947, 0.2939292788505554, 0.3211876153945923, 0.09090942144393921, 0.10683024674654007, 0.3601209819316864, 0.3663609027862549, 0.4036162197589874], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8cd90e888148fa96de94a7526cd82b21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.25599902868270874, 0.1396738439798355, 0.362801194190979, 0.39548158645629883, 0.25889021158218384, 0.2006731778383255, 0.4167245030403137, 0.4161047339439392, 0.1630173772573471, 0.3588005006313324, 0.273106187582016, 0.05884767696261406, 0.017136884853243828, 0.18808259069919586, 0.32606056332588196, 0.47483059763908386, 0.2074398696422577, 0.2892627418041229, 0.4575323462486267, 0.13849958777427673, 0.36717066168785095, 0.34571149945259094, 0.3694801926612854, 0.37464144825935364], dtype='float32').reshape([24]),
            paddle.to_tensor([0.3591902256011963, 0.016196629032492638, 0.15209847688674927, 0.3782707750797272, 0.08604823052883148, 0.10876551270484924, 0.3035818934440613, 0.34033504128456116, 0.3719601631164551, 0.18991534411907196, 0.2965843379497528, 0.20926766097545624, 0.1343764364719391, 0.3494398891925812, 0.06425300240516663, 0.18336115777492523, 0.08322586864233017, 0.38804125785827637, 0.23023860156536102, 0.3294859528541565, 0.2018568068742752, 0.24597583711147308, 0.11924022436141968, 0.42048680782318115], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f9826520cb9f04707e9c0046ebe9426d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.40099185705184937, 0.2456011176109314, 0.12352950870990753, 0.25216856598854065, 0.2630184292793274, 0.3151211440563202, 0.4328896105289459, 0.05237346887588501, 0.38539284467697144, 0.4673290252685547, 0.09777123481035233, 0.38350048661231995, 0.09805149585008621, 0.006997129879891872, 0.3231181502342224, 0.46741804480552673, 0.4412451386451721, 0.3880082666873932, 0.005674261599779129, 0.39295679330825806, 0.18637141585350037, 0.14476168155670166, 0.14057494699954987, 0.4187673032283783], dtype='float32').reshape([24]),
            paddle.to_tensor([0.4721943736076355, 0.22462841868400574, 0.45462802052497864, 0.3265056908130646, 0.44447705149650574, 0.4873240888118744, 0.3870282769203186, 0.2445104867219925, 0.20599457621574402, 0.29402902722358704, 0.3615923523902893, 0.1600436121225357, 0.12836667895317078, 0.23290058970451355, 0.3609430193901062, 0.049826480448246, 0.2798941433429718, 0.4837586283683777, 0.3053629994392395, 0.24311698973178864, 0.21264836192131042, 0.3632981479167938, 0.2687608301639557, 0.23331208527088165], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b1f629b808a0bccbe6f714f4d937c1ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.15916624665260315, 0.30129116773605347, 0.4341990649700165, 0.08781782537698746, 0.35491085052490234, 0.3393780589103699, 0.04864159971475601, 0.1711697280406952, 0.18672475218772888, 0.3324241638183594, 0.17518575489521027, 0.26572081446647644, 0.3868301510810852, 0.18951106071472168, 0.1197492927312851, 0.06457968801259995, 0.38196906447410583, 0.08795572817325592, 0.12297769635915756, 0.439588338136673, 0.2740953862667084, 0.29074352979660034, 0.11280107498168945, 0.3202146291732788], dtype='float32').reshape([24]),
            paddle.to_tensor([0.03591357171535492, 0.3915875554084778, 0.3049783706665039, 0.45237651467323303, 0.2766997516155243, 0.07701076567173004, 0.101206474006176, 0.2556135952472687, 0.2429269403219223, 0.38586580753326416, 0.3399960994720459, 0.2913559079170227, 0.3584853708744049, 0.237050861120224, 0.46640413999557495, 0.209255188703537, 0.038706108927726746, 0.05576273053884506, 0.11080672591924667, 0.343860000371933, 0.3634796440601349, 0.48467907309532166, 0.023275084793567657, 0.03292467072606087], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b22ca447343cb4b696e89f708c92345c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.038909077644348145, 0.3399308919906616, 0.3942354917526245, 0.024993911385536194, 0.23892943561077118, 0.4856584072113037, 0.12973549962043762, 0.11979029327630997, 0.10715901851654053, 0.09510361403226852, 0.025557542219758034, 0.34125566482543945, 0.3229012191295624, 0.09680626541376114, 0.05019602179527283, 0.08694581687450409, 0.27681317925453186, 0.2181738317012787, 0.10523960739374161, 0.3929179608821869, 0.19292506575584412, 0.1268174648284912, 0.10058706998825073, 0.039422422647476196], dtype='float32').reshape([24]),
            paddle.to_tensor([0.04009241238236427, 0.4719999134540558, 0.05338800325989723, 0.2771954834461212, 0.22848385572433472, 0.3336256742477417, 0.23608368635177612, 0.097867950797081, 0.3537161648273468, 0.04821527749300003, 0.48825299739837646, 0.08220426738262177, 0.3223990797996521, 0.1440141201019287, 0.2643798887729645, 0.08770913630723953, 0.11175554990768433, 0.3620840013027191, 0.40019500255584717, 0.3350338339805603, 0.37404751777648926, 0.3628470301628113, 0.14210949838161469, 0.05353432521224022], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_96f514c74253d8b03fec6a212e3cdc76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1574077159166336, 0.3334123492240906, 0.09426672011613846, 0.06949065625667572, 0.4422285556793213, 0.315021276473999, 0.1329808533191681, 0.383741170167923, 0.4856407046318054, 0.4788067936897278, 0.10565251111984253, 0.4423671364784241, 0.05679147690534592, 0.2548207640647888, 0.13908345997333527, 0.06867428869009018, 0.4008411169052124, 0.05972125753760338, 0.03979163244366646, 0.16385677456855774, 0.024615496397018433, 0.15653987228870392, 0.1204904094338417, 0.3101099729537964], dtype='float32').reshape([24]),
            paddle.to_tensor([0.18327464163303375, 0.2106115221977234, 0.4599856436252594, 0.2389131784439087, 0.39463287591934204, 0.2049688994884491, 0.35892537236213684, 0.11915910243988037, 0.06872140616178513, 0.24907539784908295, 0.377062052488327, 0.3357710838317871, 0.3412764072418213, 0.038652289658784866, 0.10039730370044708, 0.2673295736312866, 0.18875476717948914, 0.10459449142217636, 0.3070310652256012, 0.4416041374206543, 0.3604715168476105, 0.15391291677951813, 0.15516453981399536, 0.25252071022987366], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e2396287d7181a161b8186409be3a92b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.04878520593047142, 0.17960986495018005, 0.49700823426246643, 0.23891982436180115, 0.17502808570861816, 0.4426323175430298, 0.05405236780643463, 0.015351681038737297, 0.27742770314216614, 0.08839289844036102, 0.001497834688052535, 0.44509416818618774, 0.3125479817390442, 0.22210834920406342, 0.0676148384809494, 0.20860689878463745, 0.22676384449005127, 0.39132800698280334, 0.11458592116832733, 0.03387640789151192, 0.37026408314704895, 0.22858281433582306, 0.08239870518445969, 0.21815472841262817], dtype='float32').reshape([24]),
            paddle.to_tensor([0.06007709354162216, 0.053530313074588776, 0.28479912877082825, 0.11635381728410721, 0.2747719883918762, 0.07908884435892105, 0.423241525888443, 0.3920823633670807, 0.08847576379776001, 0.11656481772661209, 0.2965747117996216, 0.20334912836551666, 0.17061574757099152, 0.04140928015112877, 0.478849321603775, 0.2084011733531952, 0.16306337714195251, 0.08906558156013489, 0.179595485329628, 0.3444235622882843, 0.00749715231359005, 0.18442001938819885, 0.05504681169986725, 0.17166440188884735], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_87c737696dfb3d3e3baef51f72794760(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3719020187854767, 0.16279612481594086, 0.13391251862049103, 0.3414532542228699, 0.07134281098842621, 0.381379097700119, 0.21372012794017792, 0.12899459898471832, 0.018952375277876854, 0.24479302763938904, 0.4289688467979431, 0.45822930335998535, 0.2134883999824524, 0.11343982815742493, 0.13292600214481354, 0.09579435735940933, 0.41478431224823, 0.14802493155002594, 0.24160359799861908, 0.18166586756706238, 0.4009978771209717, 0.1671902984380722, 0.19765590131282806, 0.017324866726994514], dtype='float32').reshape([24]),
            paddle.to_tensor([0.2767189145088196, 0.485084593296051, 0.11646454781293869, 0.1308886706829071, 0.14616422355175018, 0.28169798851013184, 0.1651478409767151, 0.1612265259027481, 0.34223508834838867, 0.4733552634716034, 0.04274512827396393, 0.32972681522369385, 0.20572535693645477, 0.15287120640277863, 0.07068713009357452, 0.36129626631736755, 0.12649084627628326, 0.15651515126228333, 0.21436689794063568, 0.27790597081184387, 0.47011566162109375, 0.2592821717262268, 0.04462004080414772, 0.21350638568401337], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a0fda796ec0b15d90d04fb1bf75ee5c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2547762095928192, 0.3498023748397827, 0.3156878650188446, 0.4879373013973236, 0.2528179883956909, 0.32179269194602966, 0.38072264194488525, 0.2935139536857605, 0.04912817105650902, 0.395313024520874, 0.08545083552598953, 0.14934542775154114, 0.13591080904006958, 0.06065186485648155, 0.3273535370826721, 0.14702588319778442, 0.014861243776977062, 0.39115723967552185, 0.1579233705997467, 0.3674466907978058, 0.21116913855075836, 0.3814476430416107, 0.38902825117111206, 0.4511795938014984], dtype='float32').reshape([24]),
            paddle.to_tensor([0.1958245486021042, 0.18595807254314423, 0.46121615171432495, 0.21722836792469025, 0.4886452555656433, 0.3589939475059509, 0.28649839758872986, 0.4112842381000519, 0.17734384536743164, 0.11697845160961151, 0.3101177513599396, 0.41952264308929443, 0.12558411061763763, 0.4209752678871155, 0.4695117771625519, 0.28637826442718506, 0.09432416409254074, 0.07567030191421509, 0.21127890050411224, 0.10766322165727615, 0.20428743958473206, 0.4794478118419647, 0.45492061972618103, 0.3151381015777588], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4325107a617a35facdd3fc292787b6a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.48980507254600525, 0.20225262641906738, 0.04057382047176361, 0.33090081810951233, 0.2324766218662262, 0.15127529203891754, 0.3132180869579315, 0.45078277587890625, 0.09861063212156296, 0.16340070962905884, 0.31645265221595764, 0.4395062029361725, 0.42577478289604187, 0.24138735234737396, 0.41486769914627075, 0.16197338700294495, 0.4532359838485718, 0.16531799733638763, 0.4776124358177185, 0.2765220105648041, 0.2915506958961487, 0.010984274558722973, 0.344181627035141, 0.1449827253818512], dtype='float32').reshape([24]),
            paddle.to_tensor([0.2537020742893219, 0.4351765215396881, 0.40081095695495605, 0.29641634225845337, 0.08549311012029648, 0.4368189871311188, 0.33629897236824036, 0.03362615406513214, 0.47051942348480225, 0.34780511260032654, 0.2041577398777008, 0.3109087646007538, 0.04840646684169769, 0.3595603406429291, 0.09749352186918259, 0.11423201113939285, 0.46732190251350403, 0.42387592792510986, 0.4630972445011139, 0.1514451801776886, 0.3374304175376892, 0.32001638412475586, 0.4781065583229065, 0.28126949071884155], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c324d32e108d62ea24e1befa2862dbe6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2204131782054901, 0.03807526081800461, 0.48926469683647156, 0.19799785315990448, 0.3907511234283447, 0.11003187298774719, 0.2395654171705246, 0.29793038964271545, 0.0872715637087822, 0.07126103341579437, 0.4106282591819763, 0.29929354786872864, 0.49663305282592773, 0.43654561042785645, 0.29005733132362366, 0.455079048871994, 0.3735303580760956, 0.0888025313615799, 0.33488884568214417, 0.27772805094718933, 0.15267568826675415, 0.1849013864994049, 0.06742595136165619, 0.4781872630119324], dtype='float32').reshape([24]),
            paddle.to_tensor([0.44875311851501465, 0.06154567748308182, 0.38379648327827454, 0.06762097775936127, 0.48407137393951416, 0.3723605275154114, 0.36683350801467896, 0.04954904690384865, 0.02068895660340786, 0.024976611137390137, 0.06447148323059082, 0.2950979173183441, 0.2849016785621643, 0.07215218991041183, 0.41507527232170105, 0.44508618116378784, 0.1184140145778656, 0.27914223074913025, 0.41455212235450745, 0.12697471678256989, 0.28466543555259705, 0.3009752035140991, 0.47303274273872375, 0.01803208328783512], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_65ef668d6efc3a3629191fa2323b4ac4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2516383230686188, 0.047756925225257874, 0.03063337877392769, 0.11361321061849594, 0.24711111187934875, 0.08640994131565094, 0.2960216999053955, 0.22944676876068115, 0.3228716552257538, 0.3874753713607788, 0.21958103775978088, 0.13152103126049042, 0.007129548117518425, 0.445732980966568, 0.2601751387119293, 0.07645515352487564, 0.05519404262304306, 0.2225545048713684, 0.13739489018917084, 0.499168336391449, 0.04905478283762932, 0.4859500825405121, 0.3245089054107666, 0.06479968875646591], dtype='float32').reshape([24]),
            paddle.to_tensor([0.19572697579860687, 0.38306108117103577, 0.10550160706043243, 0.030066341161727905, 0.03104914352297783, 0.3735658824443817, 0.3557254374027252, 0.09984175860881805, 0.4012469947338104, 0.12817606329917908, 0.09454776346683502, 0.18820486962795258, 0.4995112419128418, 0.10901344567537308, 0.2803332805633545, 0.3449530601501465, 0.4299734830856323, 0.07380348443984985, 0.07215210795402527, 0.41033700108528137, 0.46160903573036194, 0.3456532955169678, 0.19057802855968475, 0.4349082410335541], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ec94d8575386a159dc24ec91050ab0b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.25554725527763367, 0.43168118596076965, 0.09809330850839615, 0.033004239201545715, 0.10878163576126099, 0.19198276102542877, 0.48066094517707825, 0.2239132970571518, 0.2479427456855774, 0.11032247543334961, 0.24195076525211334, 0.16153396666049957, 0.16218096017837524, 0.34178224205970764, 0.09137929975986481, 0.012891361489892006, 0.144174724817276, 0.4792724549770355, 0.3658965826034546, 0.21254576742649078, 0.27759334444999695, 0.02020539529621601, 0.3824334442615509, 0.08955428004264832], dtype='float32').reshape([24]),
            paddle.to_tensor([0.4210211932659149, 0.17378754913806915, 0.4390000104904175, 0.44076141715049744, 0.43140843510627747, 0.3801531493663788, 0.42686688899993896, 0.23885808885097504, 0.46993622183799744, 0.18452495336532593, 0.27816128730773926, 0.017498958855867386, 0.4972798526287079, 0.4798513352870941, 0.12299314141273499, 0.04091224819421768, 0.49109983444213867, 0.10709208250045776, 0.3041246831417084, 0.08253483474254608, 0.32801520824432373, 0.48760688304901123, 0.2133154571056366, 0.08507134765386581], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_59a38480bb106cd2ba5d360033bb892d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3663907051086426, 0.35455647110939026, 0.2186216115951538, 0.08294322341680527, 0.06054316461086273, 0.1406097412109375, 0.11310704052448273, 0.16483837366104126, 0.04879584163427353, 0.11783308535814285, 0.13864853978157043, 0.29854729771614075, 0.31629136204719543, 0.41068923473358154, 0.3913551867008209, 0.29904869198799133, 0.3528215289115906, 0.44186416268348694, 0.4991440773010254, 0.28836682438850403, 0.2559347152709961, 0.1443294733762741, 0.007923737168312073, 0.39657965302467346], dtype='float32').reshape([24]),
            paddle.to_tensor([0.43868494033813477, 0.053570397198200226, 0.28789588809013367, 0.0493122860789299, 0.27466684579849243, 0.47885391116142273, 0.35287562012672424, 0.19078174233436584, 0.4644488990306854, 0.31196051836013794, 0.14379744231700897, 0.022491516545414925, 0.48410195112228394, 0.0745403915643692, 0.4248199462890625, 0.18228530883789062, 0.4969000816345215, 0.2985125780105591, 0.4252690076828003, 0.11471006274223328, 0.22579613327980042, 0.2814078629016876, 0.24145296216011047, 0.4465121626853943], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d276d5803c0185404ecf481d7d3fa7c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4543927013874054, 0.2887659966945648, 0.2771182656288147, 0.16910862922668457, 0.22770319879055023, 0.22207997739315033, 0.3673876225948334, 0.03923211991786957, 0.12822340428829193, 0.002295473124831915, 0.13044649362564087, 0.3756214380264282, 0.3156859278678894, 0.44269824028015137, 0.30580219626426697, 0.15461517870426178, 0.24539262056350708, 0.026282576844096184, 0.26973599195480347, 0.29480624198913574, 0.1261141449213028, 0.14105457067489624, 0.3859430253505707, 0.4714984893798828], dtype='float32').reshape([24]),
            paddle.to_tensor([0.4200001657009125, 0.37605801224708557, 0.3129108250141144, 0.2710762321949005, 0.06828773766756058, 0.12413162738084793, 0.1646467000246048, 0.11383646726608276, 0.14136642217636108, 0.40018269419670105, 0.17496038973331451, 0.3614141047000885, 0.23492781817913055, 0.2573397159576416, 0.17333728075027466, 0.01121706422418356, 0.3146924674510956, 0.18423867225646973, 0.40561532974243164, 0.06707312911748886, 0.4913322925567627, 0.020940247923135757, 0.1494779884815216, 0.05654345825314522], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f5260d74a7f3bba1c6ae7ba54d223497(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.36723625659942627, 0.4219793975353241, 0.2395118921995163, 0.39058825373649597, 0.21673628687858582, 0.04407937079668045, 0.15127533674240112, 0.20488286018371582, 0.1764059215784073, 0.4978099763393402, 0.03785005211830139, 0.1496284008026123, 0.444339394569397, 0.3051510751247406, 0.32262101769447327, 0.2426222264766693, 0.17632147669792175, 0.12882472574710846, 0.48677319288253784, 0.39660948514938354, 0.06599558144807816, 0.001164985354989767, 0.41827720403671265, 0.06179322674870491], dtype='float32').reshape([24]),
            paddle.to_tensor([0.1887260377407074, 0.09598851203918457, 0.3264661729335785, 0.4065801203250885, 0.38987240195274353, 0.1903878003358841, 0.1631097048521042, 0.300360769033432, 0.3441500961780548, 0.06505870819091797, 0.33313700556755066, 0.04649403318762779, 0.4551711976528168, 0.1384315937757492, 0.3590712249279022, 0.11441738903522491, 0.37692493200302124, 0.2756788730621338, 0.23474092781543732, 0.3535337746143341, 0.29002466797828674, 0.4498613774776459, 0.18530374765396118, 0.01379961147904396], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_447672839ffc9a81d2d135ba56597535(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2161395251750946, 0.3661425709724426, 0.46938997507095337, 0.0909985825419426, 0.36618661880493164, 0.40171000361442566, 0.2434929758310318, 0.21525532007217407, 0.011513223871588707, 0.035295430570840836, 0.3922966718673706, 0.34940701723098755, 0.03880881518125534, 0.08333639055490494, 0.14734575152397156, 0.3802799582481384, 0.058920178562402725, 0.4193512201309204, 0.2635982632637024, 0.4041859805583954, 0.46978452801704407, 0.284390389919281, 0.42282527685165405, 0.3870491683483124], dtype='float32').reshape([24]),
            paddle.to_tensor([0.03583795204758644, 0.16653122007846832, 0.004321552813053131, 0.34538742899894714, 0.0026336314622312784, 0.3622661828994751, 0.08812008053064346, 0.48882657289505005, 0.09329172223806381, 0.3659762740135193, 0.009717301465570927, 0.03383651748299599, 0.07134215533733368, 0.2950221002101898, 0.3951059877872467, 0.33725982904434204, 0.21132217347621918, 0.07368962466716766, 0.08601662516593933, 0.0923643484711647, 0.12226413935422897, 0.2507210373878479, 0.39674901962280273, 0.08643220365047455], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_19d0fcf0b3cae3e6c68d8056c02fb56e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2365260273218155, 0.45617425441741943, 0.4253959059715271, 0.2381434589624405, 0.12551376223564148, 0.4907849133014679, 0.24250657856464386, 0.12615105509757996, 0.040647100657224655, 0.12364145368337631, 0.17677100002765656, 0.34418466687202454, 0.37832024693489075, 0.01046819519251585, 0.15602649748325348, 0.16724339127540588, 0.49050188064575195, 0.46733924746513367, 0.19932927191257477, 0.029527762904763222, 0.30439087748527527, 0.27832484245300293, 0.29128706455230713, 0.21580353379249573], dtype='float32').reshape([24]),
            paddle.to_tensor([0.1150556430220604, 0.27425748109817505, 0.35669612884521484, 0.4079514145851135, 0.21220143139362335, 0.1703212410211563, 0.053285904228687286, 0.4234203100204468, 0.20476634800434113, 0.35055646300315857, 0.2255714386701584, 0.39724722504615784, 0.05438618361949921, 0.4757373034954071, 0.2652585506439209, 0.08283765614032745, 0.4978693127632141, 0.26089954376220703, 0.4675462543964386, 0.1152849942445755, 0.4780542552471161, 0.48625507950782776, 0.47919830679893494, 0.1496540755033493], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8a33d5036559d25f09883d2f9b1ef4d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.39540114998817444, 0.4052145183086395, 0.24481157958507538, 0.38127702474594116, 0.26841700077056885, 0.34546026587486267, 0.47020867466926575, 0.3479892611503601, 0.14807449281215668, 0.46927914023399353, 0.19036991894245148, 0.4771966338157654, 0.191911980509758, 0.36249521374702454, 0.1571909487247467, 0.1916063278913498, 0.06131294369697571, 0.155577152967453, 0.4286286234855652, 0.25168806314468384, 0.17827445268630981, 0.3215316832065582, 0.16960911452770233, 0.13326044380664825], dtype='float32').reshape([24]),
            paddle.to_tensor([0.012689746916294098, 0.4205149710178375, 0.4720805883407593, 0.1487337201833725, 0.32790321111679077, 0.4725497364997864, 0.4298741817474365, 0.4038618505001068, 0.1413702815771103, 0.3441394567489624, 0.29578638076782227, 0.15150883793830872, 0.12163267284631729, 0.12499600648880005, 0.2777296304702759, 0.265295147895813, 0.14899104833602905, 0.2085043489933014, 0.10016706585884094, 0.3347032070159912, 0.4612159729003906, 0.33212345838546753, 0.49622848629951477, 0.22084583342075348], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6b9673e7d4e928c3d0faf0f81096aaa1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.20074956119060516, 0.006004451774060726, 0.28077852725982666, 0.4546239376068115, 0.21417616307735443, 0.24597641825675964, 0.15045066177845, 0.06943879276514053, 0.15041644871234894, 0.4118548631668091, 0.317017525434494, 0.1997939497232437, 0.05939178913831711, 0.28186464309692383, 0.1327451765537262, 0.07069694995880127, 0.3132239580154419, 0.045049991458654404, 0.3539958596229553, 0.21250015497207642, 0.37754812836647034, 0.4742584228515625, 0.38393500447273254, 0.30397599935531616], dtype='float32').reshape([24]),
            paddle.to_tensor([0.2569807767868042, 0.3975311517715454, 0.3740795850753784, 0.3761407136917114, 0.4173624813556671, 0.26201045513153076, 0.2952626943588257, 0.43905067443847656, 0.3372174799442291, 0.3868354856967926, 0.20501548051834106, 0.0402902327477932, 0.47070422768592834, 0.37468600273132324, 0.3539433479309082, 0.08296366035938263, 0.49367520213127136, 0.4866485893726349, 0.23137977719306946, 0.40090855956077576, 0.3055494725704193, 0.3542640507221222, 0.10364445298910141, 0.4668697714805603], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_544827f1b19956da7a3302bafd98ec67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.24436210095882416, 0.1577892005443573, 0.32023486495018005, 0.4311057925224304, 0.2293730527162552, 0.14441902935504913, 0.19578328728675842, 0.350273460149765, 0.03376411274075508, 0.2338382601737976, 0.04354596138000488, 0.3282777667045593, 0.2418089359998703, 0.21038350462913513, 0.08402062207460403, 0.4877525269985199, 0.1305840164422989, 0.4571300745010376, 0.40948811173439026, 0.3138332962989807, 0.23291721940040588, 0.06096450239419937, 0.4724157750606537, 0.11062170565128326], dtype='float32').reshape([24]),
            paddle.to_tensor([0.08481891453266144, 0.39226698875427246, 0.20942898094654083, 0.4168529510498047, 0.134723961353302, 0.4427735507488251, 0.006096470169723034, 0.13637667894363403, 0.34508833289146423, 0.29255610704421997, 0.07818951457738876, 0.3503652811050415, 0.04966562241315842, 0.21193158626556396, 0.3886560797691345, 0.23403300344944, 0.08357207477092743, 0.22580283880233765, 0.32151517271995544, 0.46335485577583313, 0.36000823974609375, 0.33015701174736023, 0.46079301834106445, 0.42538461089134216], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f8c41c831061e65290d4265d95df8ccb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4664541780948639, 0.007506831083446741, 0.38206496834754944, 0.017153361812233925, 0.017645400017499924, 0.464738667011261, 0.3660934865474701, 0.26948437094688416, 0.4750320017337799, 0.4335857331752777, 0.3793811500072479, 0.36705541610717773, 0.08372177183628082, 0.4025338292121887, 0.48793432116508484, 0.06428349763154984, 0.11960878223180771, 0.46028637886047363, 0.10398127138614655, 0.11168761551380157, 0.24067647755146027, 0.4723697304725647, 0.2210990935564041, 0.46436184644699097], dtype='float32').reshape([24]),
            paddle.to_tensor([0.21702764928340912, 0.22689451277256012, 0.17190660536289215, 0.11031419038772583, 0.08126579970121384, 0.32189399003982544, 0.2388981282711029, 0.34182074666023254, 0.43248140811920166, 0.42777159810066223, 0.014244680292904377, 0.00933239609003067, 0.2599926292896271, 0.29693177342414856, 0.46707698702812195, 0.0014438708312809467, 0.2947554588317871, 0.24876250326633453, 0.1448405534029007, 0.25375500321388245, 0.12856200337409973, 0.16168108582496643, 0.055015191435813904, 0.16082388162612915], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_62ac36244d7df1add161b9cce88c0125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.41944703459739685, 0.1386147290468216, 0.4366065561771393, 0.07455060631036758, 0.0120164779946208, 0.2896839678287506, 0.4581480920314789, 0.04055575653910637, 0.34591144323349, 0.326579749584198, 0.41891586780548096, 0.23125995695590973, 0.18032914400100708, 0.3510279059410095, 0.2586159110069275, 0.0994378849864006, 0.48573851585388184, 0.26900312304496765, 0.37603023648262024, 0.04380328953266144, 0.03939492627978325, 0.10328290611505508, 0.31529170274734497, 0.25040847063064575], dtype='float32').reshape([24]),
            paddle.to_tensor([0.24319596588611603, 0.45159703493118286, 0.03518538922071457, 0.47034338116645813, 0.39459022879600525, 0.2125380039215088, 0.34677252173423767, 0.28210514783859253, 0.3646690547466278, 0.15016362071037292, 0.01984117180109024, 0.4635527431964874, 0.3094014823436737, 0.4907504916191101, 0.1598355770111084, 0.03902144357562065, 0.41641244292259216, 0.17608290910720825, 0.1316220462322235, 0.0702786073088646, 0.42263975739479065, 0.4934868812561035, 0.17756617069244385, 0.34627804160118103], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0f0596dccc5eca1f9b9c7233810ab373(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3861091136932373, 0.38152921199798584, 0.44245272874832153, 0.35791584849357605, 0.3918468952178955, 0.061743754893541336, 0.1058834120631218, 0.4484610855579376, 0.15894366800785065, 0.13482604920864105, 0.11903544515371323, 0.07469668239355087, 0.01579664833843708, 0.05232267081737518, 0.0075988261960446835, 0.18294121325016022, 0.32161077857017517, 0.37411001324653625, 0.23486994206905365, 0.27800214290618896, 0.20532283186912537, 0.40364691615104675, 0.2356698364019394, 0.03704938292503357], dtype='float32').reshape([24]),
            paddle.to_tensor([0.19327662885189056, 0.2947341501712799, 0.44033724069595337, 0.07580437511205673, 0.023780709132552147, 0.2074434459209442, 0.42227211594581604, 0.32645970582962036, 0.19513602554798126, 0.3723924160003662, 0.16821089386940002, 0.4604695737361908, 0.2886887788772583, 0.269183486700058, 0.3298802375793457, 0.19749687612056732, 0.28431466221809387, 0.08667593449354172, 0.2036309689283371, 0.1638525426387787, 0.2755579650402069, 0.41632065176963806, 0.4455493986606598, 0.1434260904788971], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0926d00a55d7aff76e8d4cf69c735576(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2928314507007599, 0.35736826062202454, 0.3843972086906433, 0.2649247348308563, 0.14423112571239471, 0.03819414600729942, 0.0928710401058197, 0.28775978088378906, 0.4608475863933563, 0.2693670690059662, 0.3294244408607483, 0.344162255525589, 0.37784212827682495, 0.16462473571300507, 0.06340683996677399, 0.42480865120887756, 0.2811238169670105, 0.45677798986434937, 0.37028244137763977, 0.29063403606414795, 0.2464531660079956, 0.3579019606113434, 0.0655788779258728, 0.35769593715667725], dtype='float32').reshape([24]),
            paddle.to_tensor([0.18442419171333313, 0.022861553356051445, 0.4484763443470001, 0.09234350174665451, 0.3241155743598938, 0.43141359090805054, 0.4819016754627228, 0.22271589934825897, 0.4861607253551483, 0.3105262815952301, 0.26027345657348633, 0.38955193758010864, 0.3826269507408142, 0.2974110245704651, 0.19895729422569275, 0.3816852569580078, 0.11880843341350555, 0.02665766142308712, 0.43944481015205383, 0.45559701323509216, 0.213868647813797, 0.4013108015060425, 0.060752253979444504, 0.342435747385025], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_be3b0ec9138528d529515192bfa52a22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.29382506012916565, 0.15121017396450043, 0.021513985469937325, 0.2047869861125946, 0.1828564703464508, 0.2566147446632385, 0.11570632457733154, 0.3820118010044098, 0.389024555683136, 0.18201781809329987, 0.4405120313167572, 0.4682217836380005, 0.051292434334754944, 0.13645558059215546, 0.49361974000930786, 0.18465663492679596, 0.2594228684902191, 0.22980239987373352, 0.18726837635040283, 0.4330078065395355, 0.22184620797634125, 0.44827374815940857, 0.46965956687927246, 0.07652278989553452], dtype='float32').reshape([24]),
            paddle.to_tensor([0.43856486678123474, 0.0906871110200882, 0.24827851355075836, 0.06989187747240067, 0.3102608919143677, 0.4121992588043213, 0.012084610760211945, 0.3237530589103699, 0.1812833696603775, 0.16134187579154968, 0.09543397277593613, 0.3766065537929535, 0.03647255524992943, 0.44298118352890015, 0.08183704316616058, 0.30951303243637085, 0.2864041030406952, 0.3509060740470886, 0.121591717004776, 0.4996446967124939, 0.171026349067688, 0.1457144021987915, 0.3155284821987152, 0.11363870650529861], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_999952c6ee05d046f5ca5e82721dda46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.395257830619812, 0.4812893271446228, 0.4035157561302185, 0.4596901535987854, 0.06588728725910187, 0.09194746613502502, 0.1580587774515152, 0.494392991065979, 0.3764691948890686, 0.3706943988800049, 0.46913158893585205, 0.21274416148662567, 0.3113678991794586, 0.49711745977401733, 0.377505898475647, 0.49972251057624817, 0.2703377604484558, 0.32867109775543213, 0.0361931212246418, 0.1302865892648697, 0.28577715158462524, 0.18079304695129395, 0.17841191589832306, 0.2649787962436676], dtype='float32').reshape([24]),
            paddle.to_tensor([0.18403877317905426, 0.06950866430997849, 0.029144462198019028, 0.08015970140695572, 0.28491443395614624, 0.34381425380706787, 0.12337134778499603, 0.27112841606140137, 0.22037489712238312, 0.14318911731243134, 0.2804096043109894, 0.058468129485845566, 0.4570331573486328, 0.49772459268569946, 0.4822782576084137, 0.096678227186203, 0.38325825333595276, 0.19170266389846802, 0.388616144657135, 0.32544660568237305, 0.19152146577835083, 0.017113853245973587, 0.016243204474449158, 0.4025624692440033], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5374c2832ee01f535ecce61a3ca4aa58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.053806185722351074, 0.45891696214675903, 0.24312211573123932, 0.46730852127075195, 0.3346273899078369, 0.3496006429195404, 0.41284483671188354, 0.03566724807024002, 0.07910394668579102, 0.43081507086753845, 0.07702542841434479, 0.39681240916252136, 0.12910319864749908, 0.48178085684776306, 0.06116705760359764, 0.24829423427581787, 0.1831100583076477, 0.06923554837703705, 0.20056380331516266, 0.36568862199783325, 0.18942758440971375, 0.22129420936107635, 0.39158546924591064, 0.3310197591781616], dtype='float32').reshape([24]),
            paddle.to_tensor([0.2112686038017273, 0.03891182690858841, 0.2521170973777771, 0.4553987681865692, 0.4649070203304291, 0.003925838507711887, 0.10582482069730759, 0.17880779504776, 0.1761930286884308, 0.114957794547081, 0.25098395347595215, 0.2860599458217621, 0.12872907519340515, 0.4109175205230713, 0.32581230998039246, 0.18088020384311676, 0.45978832244873047, 0.20228692889213562, 0.29265594482421875, 0.020767753943800926, 0.3754936158657074, 0.24707840383052826, 0.2282084971666336, 0.12432954460382462], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_22f8ddb18d628230adc3989217e79410(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.33339065313339233, 0.44183942675590515, 0.1663869470357895, 0.06813167035579681, 0.005714530125260353, 0.25634947419166565, 0.06586265563964844, 0.34697291254997253, 0.2876550853252411, 0.493533194065094, 0.4655020534992218, 0.1096431091427803, 0.3993321359157562, 0.025697045028209686, 0.2377190738916397, 0.134072408080101, 0.4011228084564209, 0.23941615223884583, 0.16872653365135193, 0.1597764790058136, 0.012867043726146221, 0.4758508503437042, 0.1426258236169815, 0.49743106961250305], dtype='float32').reshape([24]),
            paddle.to_tensor([0.24940602481365204, 0.13710568845272064, 0.19387346506118774, 0.22800549864768982, 0.4168940782546997, 0.05018024519085884, 0.14784198999404907, 0.4305501878261566, 0.1470329314470291, 0.16453875601291656, 0.1588393747806549, 0.17451587319374084, 0.056585341691970825, 0.14240014553070068, 0.26411300897598267, 0.35325172543525696, 0.43812692165374756, 0.28621286153793335, 0.07554935663938522, 0.26163265109062195, 0.46818089485168457, 0.2802594006061554, 0.31799694895744324, 0.0047715893015265465], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6fab902fe9eb7ff0f7a177389ac6bcce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.46141359210014343, 0.03614059090614319, 0.12426874786615372, 0.2589278519153595, 0.20482587814331055, 0.31854701042175293, 0.44225865602493286, 0.016221484169363976, 0.050999388098716736, 0.3883713483810425, 0.2552471458911896, 0.31796208024024963, 0.40085723996162415, 0.10466177761554718, 0.401383638381958, 0.34579214453697205, 0.21887338161468506, 0.4509848952293396, 0.006501639261841774, 0.11555369943380356, 0.2425161749124527, 0.2873491942882538, 0.2722403109073639, 0.22372837364673615], dtype='float32').reshape([24]),
            paddle.to_tensor([0.26650741696357727, 0.45501917600631714, 0.2576369643211365, 0.4583207368850708, 0.4981445074081421, 0.00787600502371788, 0.14057868719100952, 0.42438650131225586, 0.10331928730010986, 0.40191650390625, 0.3945758044719696, 0.4829796254634857, 0.0986485555768013, 0.47333866357803345, 0.04450336843729019, 0.006029950920492411, 0.14278480410575867, 0.36358505487442017, 0.03841891512274742, 0.2598845064640045, 0.07067080587148666, 0.3645738959312439, 0.2139510065317154, 0.3837554156780243], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_451def721ce87136df842d6e602a1737(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.18328790366649628, 0.13790543377399445, 0.12096403539180756, 0.20254115760326385, 0.42318788170814514, 0.22869712114334106, 0.41029515862464905, 0.31199973821640015, 0.00413438631221652, 0.47484201192855835, 0.23830583691596985, 0.27385443449020386, 0.3253830671310425, 0.41182297468185425, 0.49252840876579285, 0.23393718898296356, 0.0982903316617012, 0.04956307262182236, 0.00015125394565984607, 0.3625272512435913, 0.09275129437446594, 0.04547438025474548, 0.14088954031467438, 0.2993544936180115], dtype='float32').reshape([24]),
            paddle.to_tensor([0.3733654320240021, 0.4288122057914734, 0.49884992837905884, 0.1379363238811493, 0.25350549817085266, 0.033080145716667175, 0.4714765250682831, 0.27251046895980835, 0.12945646047592163, 0.2188611924648285, 0.03092116490006447, 0.11386024951934814, 0.12772217392921448, 0.2810160517692566, 0.07152555882930756, 0.22567634284496307, 0.20455937087535858, 0.2413434386253357, 0.17944936454296112, 0.3796754777431488, 0.10451704263687134, 0.17536409199237823, 0.2761671245098114, 0.20975884795188904], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a0df8b22d31d8ef60d1b41bc2741aabc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.43386515974998474, 0.41544175148010254, 0.23280422389507294, 0.2798287868499756, 0.4644755423069, 0.31365254521369934, 0.3029618561267853, 0.05555473640561104, 0.4828035235404968, 0.42244642972946167, 0.45691871643066406, 0.41013455390930176, 0.004918091464787722, 0.30703288316726685, 0.25551971793174744, 0.34249669313430786, 0.2859838008880615, 0.37832531332969666, 0.21679389476776123, 0.3733070194721222, 0.2812666893005371, 0.31351467967033386, 0.033778008073568344, 0.3509100079536438], dtype='float32').reshape([24]),
            paddle.to_tensor([0.29477599263191223, 0.178890198469162, 0.20766042172908783, 0.4733906686306, 0.2941227853298187, 0.17971163988113403, 0.4343111515045166, 0.30361542105674744, 0.46589261293411255, 0.4885789155960083, 0.1085309162735939, 0.3114018142223358, 0.2864786684513092, 0.17709586024284363, 0.20329028367996216, 0.28622037172317505, 0.4018328785896301, 0.3139972984790802, 0.33059030771255493, 0.13680186867713928, 0.2944427728652954, 0.19074587523937225, 0.38746678829193115, 0.1618432104587555], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c2ec4579a5e4f5c0e58a9274ef365c46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4312141239643097, 0.13264116644859314, 0.1605023890733719, 0.040225494652986526, 0.4392125606536865, 0.17857183516025543, 0.4638766944408417, 0.3806072771549225, 0.07603641599416733, 0.02092881128191948, 0.2647075951099396, 0.1716555505990982, 0.15668885409832, 0.29378581047058105, 0.4653826653957367, 0.39949214458465576, 0.19621102511882782, 0.4657381772994995, 0.00622948445379734, 0.048310816287994385, 0.018053891137242317, 0.04316616430878639, 0.3190076947212219, 0.10114273428916931], dtype='float32').reshape([24]),
            paddle.to_tensor([0.052838150411844254, 0.1513373702764511, 0.40824422240257263, 0.46992865204811096, 0.3140195608139038, 0.4865038990974426, 0.42771637439727783, 0.19449453055858612, 0.1553611159324646, 0.17337357997894287, 0.18066655099391937, 0.1265137940645218, 0.15326054394245148, 0.4968477487564087, 0.453971266746521, 0.4524460732936859, 0.1557072401046753, 0.46405014395713806, 0.07133635133504868, 0.08057336509227753, 0.04825890064239502, 0.43040141463279724, 0.31876319646835327, 0.08831211179494858], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4ca1c8929ac5a83517dfda4757dee5b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4691127836704254, 0.21587099134922028, 0.17382203042507172, 0.3377580940723419, 0.23200176656246185, 0.44011011719703674, 0.312603235244751, 0.44923728704452515, 0.007605915889143944, 0.22613970935344696, 0.26542726159095764, 0.19376201927661896, 0.13222761452198029, 0.38347360491752625, 0.00841373298317194, 0.09500807523727417, 0.47308796644210815, 0.3806673288345337, 0.3817708492279053, 0.2833719849586487, 0.1323445588350296, 0.48738670349121094, 0.404735803604126, 0.014653407037258148], dtype='float32').reshape([24]),
            paddle.to_tensor([0.006123330444097519, 0.3217849135398865, 0.19295702874660492, 0.0912381187081337, 0.3869273066520691, 0.026838503777980804, 0.41028082370758057, 0.09869623184204102, 0.03278975561261177, 0.06976163387298584, 0.013199491426348686, 0.2985546588897705, 0.12086675316095352, 0.28622788190841675, 0.3507384657859802, 0.25793763995170593, 0.24174971878528595, 0.32617971301078796, 0.10887044668197632, 0.3099299371242523, 0.14295899868011475, 0.16366004943847656, 0.011801715940237045, 0.23189924657344818], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6e97ca5a09399b774e139f366493fbe7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.27697497606277466, 0.345083624124527, 0.10970999300479889, 0.15074454247951508, 0.16114360094070435, 0.4411676526069641, 0.04893328994512558, 0.07032536715269089, 0.12416011840105057, 0.09644624590873718, 0.17378029227256775, 0.4338844418525696, 0.14427806437015533, 0.19335106015205383, 0.2758598029613495, 0.49349841475486755, 0.426896870136261, 0.21997860074043274, 0.40885117650032043, 0.3530758023262024, 0.14571242034435272, 0.16428513824939728, 0.2193647176027298, 0.4398437440395355], dtype='float32').reshape([24]),
            paddle.to_tensor([0.3287707269191742, 0.2933168113231659, 0.031689539551734924, 0.476423054933548, 0.4194153845310211, 0.330296128988266, 0.03839683532714844, 0.291202574968338, 0.21126292645931244, 0.2018045037984848, 0.03757752105593681, 0.44595077633857727, 0.4742007553577423, 0.3286994993686676, 0.4370025098323822, 0.10208725184202194, 0.06700240820646286, 0.38773590326309204, 0.025147220119833946, 0.13888688385486603, 0.2952336370944977, 0.39112916588783264, 0.11948025971651077, 0.2812141478061676], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1378d84f02000ba945ba3933bbc35cb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4314621388912201, 0.2626471519470215, 0.22205787897109985, 0.15510892868041992, 0.4067836105823517, 0.07495203614234924, 0.10712786018848419, 0.1160012036561966, 0.24771995842456818, 0.006325103342533112, 0.12690584361553192, 0.4916941225528717, 0.34031912684440613, 0.04771795868873596, 0.38993507623672485, 0.36269858479499817, 0.13252438604831696, 0.18508994579315186, 0.03712392598390579, 0.2665572762489319, 0.4886528551578522, 0.15469200909137726, 0.3323485851287842, 0.2218673825263977], dtype='float32').reshape([24]),
            paddle.to_tensor([0.29170775413513184, 0.17818447947502136, 0.2120242565870285, 0.07960209995508194, 0.40622198581695557, 0.03617842495441437, 0.24670298397541046, 0.05763398855924606, 0.2749381959438324, 0.2681066691875458, 0.4478433430194855, 0.27427318692207336, 0.4001559317111969, 0.21668653190135956, 0.1972852200269699, 0.1040903702378273, 0.4223109483718872, 0.47960230708122253, 0.42504045367240906, 0.09620174020528793, 0.4985962212085724, 0.3962371349334717, 0.4192229211330414, 0.3840426504611969], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eed80d94c72451dab4a0dcbb3deac5f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.45788344740867615, 0.00045639550080522895, 0.39150288701057434, 0.3667545020580292, 0.14908815920352936, 0.3052982687950134, 0.17156554758548737, 0.2143753319978714, 0.3832223415374756, 0.3176027536392212, 0.28317880630493164, 0.1692972034215927, 0.3962440490722656, 0.030912647023797035, 0.4081241190433502, 0.31854891777038574, 0.34653472900390625, 0.21547172963619232, 0.27609989047050476, 0.22581534087657928, 0.1000174731016159, 0.057527799159288406, 0.4991688132286072, 0.21601103246212006], dtype='float32').reshape([24]),
            paddle.to_tensor([0.06386272609233856, 0.08633637428283691, 0.4770280420780182, 0.09042476117610931, 0.3519243896007538, 0.3738279640674591, 0.24946142733097076, 0.2717282772064209, 0.0031764155719429255, 0.22161734104156494, 0.4693774878978729, 0.31049254536628723, 0.4913512170314789, 0.10404103994369507, 0.4772428274154663, 0.4393569529056549, 0.13143523037433624, 0.29769647121429443, 0.028460592031478882, 0.06796008348464966, 0.33599817752838135, 0.43803369998931885, 0.35787320137023926, 0.34984827041625977], dtype='float32').reshape([24]),
        ]


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