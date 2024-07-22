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
class TestPrimitiveOp_1a91b989b372821d3df3ea83d62cb38f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.18557792901992798, 0.32007303833961487, 0.18999125063419342, 0.07081961631774902, 0.382150799036026, 0.3697015345096588, 0.34371039271354675, 0.2460322380065918, 0.4014826714992523, 0.13120827078819275, 0.21110233664512634, 0.03582832217216492, 0.11002826690673828, 0.48579031229019165, 0.47055140137672424, 0.49326616525650024, 0.02975228615105152, 0.025987448170781136, 0.017184214666485786, 0.025708984583616257, 0.032728683203458786, 0.2615104913711548, 0.32695814967155457, 0.13774420320987701], dtype='float32').reshape([24]),
            paddle.to_tensor([0.12575137615203857, 0.4861892759799957, 0.15025468170642853, 0.35640373826026917, 0.47453585267066956, 0.33910202980041504, 0.4794483482837677, 0.1600324511528015, 0.11602745205163956, 0.008278917521238327, 0.02693174220621586, 0.4506690502166748, 0.2366214245557785, 0.011804630048573017, 0.18478918075561523, 0.27150222659111023, 0.46170493960380554, 0.03931446373462677, 0.1282913237810135, 0.42320048809051514, 0.2102435827255249, 0.2237933874130249, 0.15707646310329437, 0.392195463180542], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
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
class TestPrimitiveOp_5739f5ea2aad470f264917c5eb51c939(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4894232749938965, 0.39302146434783936, 0.24774377048015594, 0.3282211124897003, 0.07065164297819138, 0.44173985719680786, 0.2776108682155609, 0.3275449275970459, 0.21766872704029083, 0.4223199784755707, 0.21058465540409088, 0.25269728899002075, 0.23878751695156097, 0.46747153997421265, 0.25833195447921753, 0.22710029780864716, 0.3212806284427643, 0.4295365810394287, 0.008130498230457306, 0.4297127425670624, 0.3733919858932495, 0.014107746072113514, 0.21050621569156647, 0.010991958901286125], dtype='float32').reshape([24]),
            paddle.to_tensor([0.4324173927307129, 0.22862039506435394, 0.49999871850013733, 0.3409220576286316, 0.11399386823177338, 0.39413684606552124, 0.0726449266076088, 0.31914374232292175, 0.35738691687583923, 0.4450126886367798, 0.3715563714504242, 0.08431700617074966, 0.2281743437051773, 0.12527106702327728, 0.0037365176249295473, 0.40489575266838074, 0.43690553307533264, 0.306119441986084, 0.17036275565624237, 0.01178570743650198, 0.07605984061956406, 0.41606131196022034, 0.2228042334318161, 0.020824164152145386], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
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
class TestPrimitiveOp_efe0178a2823bdce8bf95ed2df3f5b12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.03513530269265175, 0.3028586208820343, 0.2280714362859726, 0.37589627504348755, 0.4481565058231354, 0.36293357610702515, 0.18754443526268005, 0.1812683641910553, 0.20437921583652496, 0.45626455545425415, 0.3015091121196747, 0.2065226286649704, 0.38265523314476013, 0.07587423175573349, 0.0011120333801954985, 0.046214550733566284, 0.31792333722114563, 0.24716724455356598, 0.06855440884828568, 0.04069637879729271, 0.41614317893981934, 0.12556323409080505, 0.10468371212482452, 0.3293677270412445], dtype='float32').reshape([24]),
            paddle.to_tensor([0.055061038583517075, 0.3164403736591339, 0.2714866101741791, 0.258524626493454, 0.3732859790325165, 0.36024224758148193, 0.045409683138132095, 0.39449068903923035, 0.2997606098651886, 0.26498129963874817, 0.4412654936313629, 0.2794581651687622, 0.13612616062164307, 0.42093876004219055, 0.38086020946502686, 0.42723509669303894, 0.048407308757305145, 0.25255683064460754, 0.10977859050035477, 0.16724924743175507, 0.19889523088932037, 0.20510996878147125, 0.45520102977752686, 0.3015405833721161], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
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
class TestPrimitiveOp_8f43c4c5c178a3cafd761b225cd4eaad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0009266676497645676, 0.4569169580936432, 0.1922357976436615, 0.026860874146223068, 0.012805046513676643, 0.4691333472728729, 0.11356056481599808, 0.05572505295276642, 0.31730183959007263, 0.15613949298858643, 0.040136151015758514, 0.31188076734542847, 0.46710696816444397, 0.0707920491695404, 0.17637360095977783, 0.4591725766658783, 0.03263413906097412, 0.05042111128568649, 0.0019172467291355133, 0.14436832070350647, 0.373232901096344, 0.11397235095500946, 0.05069105699658394, 0.16660988330841064], dtype='float32').reshape([24]),
            paddle.to_tensor([0.360390841960907, 0.12386021018028259, 0.3096921145915985, 0.24416986107826233, 0.23045581579208374, 0.21057748794555664, 0.03576572239398956, 0.4865342676639557, 0.42694875597953796, 0.36033546924591064, 0.02399381250143051, 0.4239684045314789, 0.30213919281959534, 0.3426682651042938, 0.1406739503145218, 0.08832024037837982, 0.33539873361587524, 0.2701224088668823, 0.19719015061855316, 0.3361166715621948, 0.09247983247041702, 0.19217103719711304, 0.351728618144989, 0.2429441511631012], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
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
class TestPrimitiveOp_1f34c0d010025c914ab67b6831252c41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2929288446903229, 0.40260937809944153, 0.46827730536460876, 0.2568630278110504, 0.21958377957344055, 0.39780861139297485, 0.09189960360527039, 0.3860931396484375, 0.006806256249547005, 0.30961287021636963, 0.4472181499004364, 0.009248164482414722, 0.30434098839759827, 0.019307266920804977, 0.1883336305618286, 0.44343698024749756, 0.1482214480638504, 0.10681938380002975, 0.24254009127616882, 0.44587817788124084, 0.21449114382266998, 0.13496658205986023, 0.38958510756492615, 0.3233550786972046], dtype='float32').reshape([24]),
            paddle.to_tensor([0.1038304939866066, 0.47783198952674866, 0.10749951750040054, 0.44453269243240356, 0.17322668433189392, 0.3538517951965332, 0.0059220134280622005, 0.3375278413295746, 0.3292349576950073, 0.16054484248161316, 0.2742246687412262, 0.09881531447172165, 0.425597608089447, 0.11965606361627579, 0.21171344816684723, 0.13014858961105347, 0.37691208720207214, 0.49213653802871704, 0.46061280369758606, 0.22359324991703033, 0.19646857678890228, 0.0014152233488857746, 0.29803889989852905, 0.47569239139556885], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a76e8ef5a6b7d908e619c8bd1f144959(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.46142858266830444, 0.470750629901886, 0.27092957496643066, 0.1454208493232727, 0.15598635375499725, 0.038589317351579666, 0.16466012597084045, 0.2982492446899414, 0.3323836326599121, 0.16400736570358276, 0.1523159146308899, 0.26609864830970764, 0.1981530487537384, 0.48086389899253845, 0.14776700735092163, 0.29769015312194824, 0.3693331778049469, 0.19759811460971832, 0.45496347546577454, 0.17112967371940613, 0.05407559499144554, 0.05871938541531563, 0.05141723155975342, 0.04344716668128967], dtype='float32').reshape([24]),
            paddle.to_tensor([0.3204044997692108, 0.1301552653312683, 0.4498136043548584, 0.07472959905862808, 0.30138176679611206, 0.031225264072418213, 0.3523406386375427, 0.3556104302406311, 0.2625509798526764, 0.08319670706987381, 0.44868937134742737, 0.47765398025512695, 0.2014644294977188, 0.4118974804878235, 0.03678882494568825, 0.042427487671375275, 0.1265745460987091, 0.15858441591262817, 0.47994405031204224, 0.3295329213142395, 0.00915518868714571, 0.37874555587768555, 0.3858391344547272, 0.2634248435497284], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1c1a1c6ba29afde803447b22758cce79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3852593004703522, 0.061754368245601654, 0.10513415932655334, 0.09169141948223114, 0.4158417582511902, 0.273942232131958, 0.07613906264305115, 0.12092604488134384, 0.42380741238594055, 0.22722835838794708, 0.49910327792167664, 0.2510910928249359, 0.28568777441978455, 0.3285799026489258, 0.1428675353527069, 0.3996749818325043, 0.37039443850517273, 0.1949162632226944, 0.05567727982997894, 0.19043688476085663, 0.4979543387889862, 0.02417675592005253, 0.4568227529525757, 0.3067842721939087], dtype='float32').reshape([24]),
            paddle.to_tensor([0.11194608360528946, 0.39028507471084595, 0.45009341835975647, 0.4008389115333557, 0.12800481915473938, 0.19818904995918274, 0.17486077547073364, 0.4536150395870209, 0.48820021748542786, 0.4709565341472626, 0.3319510221481323, 0.016767073422670364, 0.17767038941383362, 0.04683232307434082, 0.1732078343629837, 0.32106587290763855, 0.37855473160743713, 0.23165100812911987, 0.154708594083786, 0.3543301522731781, 0.22153975069522858, 0.25542330741882324, 0.1846991330385208, 0.45740818977355957], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a0a365b754b2a6fb8d3b98d03e9db2aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.31615838408470154, 0.3888537585735321, 0.12670762836933136, 0.2716168165206909, 0.2606115937232971, 0.12328989803791046, 0.030999915674328804, 0.42016199231147766, 0.28645840287208557, 0.38627946376800537, 0.3130512535572052, 0.07873406261205673, 0.13075114786624908, 0.45866015553474426, 0.14900392293930054, 0.36201339960098267, 0.1644120216369629, 0.2197045534849167, 0.0004509511054493487, 0.2886301875114441, 0.3511703908443451, 0.2191217839717865, 0.3984113335609436, 0.39675962924957275], dtype='float32').reshape([24]),
            paddle.to_tensor([0.10704578459262848, 0.18470416963100433, 0.3558475971221924, 0.2785373330116272, 0.06888729333877563, 0.03866197168827057, 0.024407977238297462, 0.09313499927520752, 0.296229749917984, 0.0155866127461195, 0.33309775590896606, 0.04949507862329483, 0.31130826473236084, 0.03375934436917305, 0.4344682991504669, 0.41972437500953674, 0.23060400784015656, 0.20581530034542084, 0.03405694290995598, 0.3890611231327057, 0.020368656143546104, 0.1023167073726654, 0.47946467995643616, 0.06333756446838379], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c926d78152f879ad0a07b3af5bbf2530(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.030688457190990448, 0.2550361752510071, 0.10495477169752121, 0.40645548701286316, 0.46929818391799927, 0.17530560493469238, 0.3389422297477722, 0.16908644139766693, 0.1982177346944809, 0.4975198209285736, 0.14448674023151398, 0.28588640689849854, 0.2764745354652405, 0.04856292903423309, 0.2095988243818283, 0.08120176941156387, 0.03200673311948776, 0.2844375967979431, 0.1565363109111786, 0.02407647669315338, 0.4372400641441345, 0.3700490891933441, 0.11595606058835983, 0.0496029257774353], dtype='float32').reshape([24]),
            paddle.to_tensor([0.11437821388244629, 0.17038650810718536, 0.48310598731040955, 0.20973831415176392, 0.3210460841655731, 0.4348059296607971, 0.2197347730398178, 0.23732231557369232, 0.11183145642280579, 0.25240206718444824, 0.45429596304893494, 0.3402089476585388, 0.3089520037174225, 0.011120880953967571, 0.13550935685634613, 0.32128268480300903, 0.39819738268852234, 0.3210846781730652, 0.21410638093948364, 0.4513459801673889, 0.49914267659187317, 0.3547615706920624, 0.19369006156921387, 0.26052728295326233], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
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
class TestPrimitiveOp_e4aa469cc70ccfafcb8a4028854c873c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.03507506474852562, 0.4726286232471466, 0.21683938801288605, 0.47727131843566895, 0.2138596922159195, 0.15568046271800995, 0.3742484152317047, 0.051243461668491364, 0.163829505443573, 0.4451495110988617, 0.14013652503490448, 0.4116010069847107, 0.12977293133735657, 0.2837769389152527, 0.1449921876192093, 0.4418068826198578, 0.19956544041633606, 0.07563529163599014, 0.30375921726226807, 0.06277606636285782, 0.4096645712852478, 0.06267563998699188, 0.39228057861328125, 0.1536305695772171], dtype='float32').reshape([24]),
            paddle.to_tensor([0.23923535645008087, 0.03189670294523239, 0.31410402059555054, 0.35864272713661194, 0.22936499118804932, 0.2988196611404419, 0.4852163791656494, 0.1667562872171402, 0.19574952125549316, 0.07073426991701126, 0.08981485664844513, 0.24961453676223755, 0.03717724233865738, 0.40135490894317627, 0.37110987305641174, 0.25225988030433655, 0.4806106984615326, 0.43387702107429504, 0.22379320859909058, 0.02887541987001896, 0.4826131761074066, 0.4258623421192169, 0.3591468930244446, 0.35681867599487305], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
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
class TestPrimitiveOp_ae79a2ee4c07a317e060a824247d334b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3134850561618805, 0.3172972798347473, 0.38479962944984436, 0.3205386698246002, 0.44504889845848083, 0.3865216374397278, 0.22452259063720703, 0.15682221949100494, 0.30840229988098145, 0.07892552018165588, 0.4569323658943176, 0.11894088983535767, 0.46890756487846375, 0.06404297798871994, 0.4316891133785248, 0.33794549107551575, 0.32232537865638733, 0.17606180906295776, 0.21438266336917877, 0.22911998629570007, 0.4944925308227539, 0.4872172474861145, 0.3546540141105652, 0.3062010407447815], dtype='float32').reshape([24]),
            paddle.to_tensor([0.23078526556491852, 0.3358755111694336, 0.14085254073143005, 0.011944515630602837, 0.29519596695899963, 0.15833447873592377, 0.05954698100686073, 0.4254775643348694, 0.46309539675712585, 0.199662983417511, 0.47849521040916443, 0.2883360981941223, 0.4451708495616913, 0.2812700569629669, 0.010555868037045002, 0.15761737525463104, 0.015374504961073399, 0.04203938692808151, 0.3856744170188904, 0.49273914098739624, 0.07863856106996536, 0.43883219361305237, 0.4896291494369507, 0.34693238139152527], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
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
class TestPrimitiveOp_0afd44299edd133ee743faa9a4dc0e58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1759195774793625, 0.13655953109264374, 0.08970408141613007, 0.4518701434135437, 0.3339511752128601, 0.4302493631839752, 0.44165804982185364, 0.19940832257270813, 0.29347583651542664, 0.13689245283603668, 0.33998721837997437, 0.331850528717041, 0.258860319852829, 0.1024511530995369, 0.0463242344558239, 0.3564761281013489, 0.0020363135263323784, 0.3181179463863373, 0.23726680874824524, 0.3712978661060333, 0.06424477696418762, 0.1054452657699585, 0.24060940742492676, 0.14433854818344116], dtype='float32').reshape([24]),
            paddle.to_tensor([0.48519012331962585, 0.4975358247756958, 0.3693622946739197, 0.04142696410417557, 0.4873191714286804, 0.1974269598722458, 0.4720301926136017, 0.26236721873283386, 0.10473328083753586, 0.2444400042295456, 0.15042518079280853, 0.334561824798584, 0.0031463196501135826, 0.16767311096191406, 0.35956871509552, 0.3548411726951599, 0.326074481010437, 0.15529417991638184, 0.11866538971662521, 0.15334293246269226, 0.2492513507604599, 0.054638683795928955, 0.45392608642578125, 0.20398391783237457], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a753a6234ffbb582f4223a1f8e9490b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.026115378364920616, 0.3399978280067444, 0.18573975563049316, 0.1830003261566162, 0.12409113347530365, 0.44811078906059265, 0.3746044337749481, 0.05691293999552727, 0.46117648482322693, 0.21032361686229706, 0.34848594665527344, 0.42765700817108154, 0.48673659563064575, 0.33184465765953064, 0.2274269312620163, 0.07443918287754059, 0.19503997266292572, 0.11396331340074539, 0.19778768718242645, 0.35529500246047974, 0.20287920534610748, 0.47535407543182373, 0.4010811746120453, 0.4136248230934143], dtype='float32').reshape([24]),
            paddle.to_tensor([0.3828301727771759, 0.44412490725517273, 0.18499040603637695, 0.3435114026069641, 0.3052401542663574, 0.3681553304195404, 0.0995011106133461, 0.01962503232061863, 0.2231067717075348, 0.4545924961566925, 0.07231321930885315, 0.4412509500980377, 0.3204094469547272, 0.3284834921360016, 0.1406443864107132, 0.041892196983098984, 0.08240841329097748, 0.17574793100357056, 0.2553008198738098, 0.079107366502285, 0.3947557806968689, 0.4868345558643341, 0.4567123353481293, 0.4347275197505951], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4a96c0e5ae5dbf4e761aa699c3cc5cbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4758933484554291, 0.14255014061927795, 0.49187612533569336, 0.047913145273923874, 0.12605196237564087, 0.4230193793773651, 0.41245803236961365, 0.07384315133094788, 0.21088989078998566, 0.34581270813941956, 0.3833952844142914, 0.2822886109352112, 0.45838046073913574, 0.16176468133926392, 0.4614314138889313, 0.2802913784980774, 0.19649264216423035, 0.1444588452577591, 0.03105638176202774, 0.23455320298671722, 0.002982559846714139, 0.44490891695022583, 0.021438896656036377, 0.4839061498641968], dtype='float32').reshape([24]),
            paddle.to_tensor([0.051319241523742676, 0.4611882269382477, 0.22770775854587555, 0.11426018178462982, 0.3777332603931427, 0.31432104110717773, 0.46817252039909363, 0.2517809271812439, 0.11959251761436462, 0.1371832937002182, 0.015778006985783577, 0.13908152282238007, 0.005648371763527393, 0.015706151723861694, 0.4778655171394348, 0.11719158291816711, 0.22941698133945465, 0.04897738993167877, 0.10249515622854233, 0.40890949964523315, 0.4858309030532837, 0.10021840035915375, 0.4068313241004944, 0.25689175724983215], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
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
class TestPrimitiveOp_f36235950b370949b48077b9234f1d81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2873746156692505, 0.1815248727798462, 0.12061513215303421, 0.32241445779800415, 0.3212587237358093, 0.28378304839134216, 0.33587872982025146, 0.4938142001628876, 0.19452793896198273, 0.2163516730070114, 0.4053199887275696, 0.05240008607506752, 0.2817927598953247, 0.1612112820148468, 0.08422444760799408, 0.19342376291751862, 0.11218459159135818, 0.056300655007362366, 0.4635474681854248, 0.025622330605983734, 0.3040720820426941, 0.38445475697517395, 0.38712674379348755, 0.4090229868888855], dtype='float32').reshape([24]),
            paddle.to_tensor([0.18952342867851257, 0.2906690835952759, 0.04539253190159798, 0.15193432569503784, 0.24614083766937256, 0.15173020958900452, 0.4556560516357422, 0.4584605395793915, 0.2738528847694397, 0.28499531745910645, 0.2899218201637268, 0.10980573296546936, 0.3619408905506134, 0.30852559208869934, 0.4138665199279785, 0.11654220521450043, 0.35846373438835144, 0.2868020236492157, 0.1309565156698227, 0.22102117538452148, 0.001892277505248785, 0.25493401288986206, 0.21538372337818146, 0.41735348105430603], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
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
class TestPrimitiveOp_b1698cc5984148d09d0c7b62d0c33386(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.009980613365769386, 0.2066096067428589, 0.052638985216617584, 0.08239495754241943, 0.008461661636829376, 0.37111377716064453, 0.11026058346033096, 0.09752855449914932, 0.2233685702085495, 0.15352359414100647, 0.23099084198474884, 0.005710940808057785, 0.4699607789516449, 0.22244541347026825, 0.08684197068214417, 0.44932010769844055, 0.10638318955898285, 0.1156611517071724, 0.25119829177856445, 0.05250447988510132, 0.12593379616737366, 0.0181891992688179, 0.05210256204009056, 0.4532413184642792], dtype='float32').reshape([24]),
            paddle.to_tensor([0.10958357900381088, 0.24095791578292847, 0.06223257631063461, 0.19261164963245392, 0.23442797362804413, 0.0922679752111435, 0.02529970370233059, 0.3026752173900604, 0.42767420411109924, 0.49469730257987976, 0.3415588438510895, 0.32167384028434753, 0.16551581025123596, 0.31206464767456055, 0.13808698952198029, 0.3272615671157837, 0.3406936526298523, 0.11879443377256393, 0.2191116213798523, 0.14108748733997345, 0.026989929378032684, 0.185555100440979, 0.4982188642024994, 0.12501585483551025], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
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
class TestPrimitiveOp_4b73bd94f9d19d0428dab53a7d0a50e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3045271337032318, 0.36088189482688904, 0.4545067548751831, 0.422355055809021, 0.42839276790618896, 0.3346051275730133, 0.009054453112185001, 0.1608443260192871, 0.2292409986257553, 0.029927683994174004, 0.03858471289277077, 0.20469410717487335, 0.1728423535823822, 0.175882488489151, 0.39274173974990845, 0.26461121439933777, 0.3576749563217163, 0.10799141228199005, 0.19293850660324097, 0.05502169951796532, 0.05014834180474281, 0.2944068908691406, 0.3989642262458801, 0.3900720179080963], dtype='float32').reshape([24]),
            paddle.to_tensor([0.35645776987075806, 0.26274392008781433, 0.21716605126857758, 0.2551792562007904, 0.34680429100990295, 0.48434537649154663, 0.20857588946819305, 0.4435752034187317, 0.4403742849826813, 0.27582046389579773, 0.18716412782669067, 0.23656408488750458, 0.3588096499443054, 0.12757307291030884, 0.39947572350502014, 0.4014561176300049, 0.3294893503189087, 0.2672031819820404, 0.16525886952877045, 0.20853517949581146, 0.43141835927963257, 0.42793288826942444, 0.45923808217048645, 0.20088329911231995], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
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
class TestPrimitiveOp_c1262b1efc3c49a37d8543349436c8e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.125257208943367, 0.26630279421806335, 0.37516918778419495, 0.06796825677156448, 0.14348039031028748, 0.432963103055954, 0.23445843160152435, 0.255221426486969, 0.03421889990568161, 0.4957563579082489, 0.3195517957210541, 0.22233279049396515, 0.340664267539978, 0.24260041117668152, 0.10078512877225876, 0.3019268810749054, 0.4324286878108978, 0.3794633150100708, 0.2903705835342407, 0.2803431749343872, 0.058363426476716995, 0.19376982748508453, 0.46515753865242004, 0.21511000394821167], dtype='float32').reshape([24]),
            paddle.to_tensor([0.0890764519572258, 0.3386397957801819, 0.35162442922592163, 0.15555673837661743, 0.22990140318870544, 0.1320718675851822, 0.15941056609153748, 0.4153274595737457, 0.4542528986930847, 0.33819878101348877, 0.1382564902305603, 0.3610631823539734, 0.24435286223888397, 0.48980000615119934, 0.36535245180130005, 0.3917272984981537, 0.18642082810401917, 0.30590805411338806, 0.43089011311531067, 0.2654983103275299, 0.46121129393577576, 0.4739340543746948, 0.16264113783836365, 0.13679340481758118], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
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
class TestPrimitiveOp_1bb3ad15e908b461c1d398abf108a1a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.22056183218955994, 0.3215359151363373, 0.46935907006263733, 0.24616269767284393, 0.022607149556279182, 0.17583952844142914, 0.45278456807136536, 0.33680304884910583, 0.28212082386016846, 0.10420001298189163, 0.16569019854068756, 0.4431293308734894, 0.38582777976989746, 0.04099467769265175, 0.24310868978500366, 0.39429986476898193, 0.02201562374830246, 0.09620098024606705, 0.29747137427330017, 0.11888834834098816, 0.4286400377750397, 0.33920061588287354, 0.1445033848285675, 0.13719695806503296], dtype='float32').reshape([24]),
            paddle.to_tensor([0.1128792092204094, 0.2592217028141022, 0.19194038212299347, 0.3717268109321594, 0.3226310908794403, 0.4229230582714081, 0.08634963631629944, 0.35607409477233887, 0.32209131121635437, 0.27355295419692993, 0.2836862802505493, 0.3129297196865082, 0.49770551919937134, 0.35394778847694397, 0.1329210102558136, 0.12742388248443604, 0.418045312166214, 0.2860706150531769, 0.29434463381767273, 0.01134207472205162, 0.4698030948638916, 0.2880990505218506, 0.15108497440814972, 0.029141156002879143], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a4cf3f7853ae9648acaa54c0d795cf22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.360409140586853, 0.08726335316896439, 0.12895233929157257, 0.35411882400512695, 0.3476443588733673, 0.4807546138763428, 0.46181726455688477, 0.43201878666877747, 0.002793261082842946, 0.1495034396648407, 0.4345085322856903, 0.2610032558441162, 0.21554253995418549, 0.059939850121736526, 0.4966655671596527, 0.45420312881469727, 0.43090003728866577, 0.0748104378581047, 0.08833171427249908, 0.3165273666381836, 0.06683310866355896, 0.34681040048599243, 0.33262521028518677, 0.16069728136062622], dtype='float32').reshape([24]),
            paddle.to_tensor([0.05348410829901695, 0.48318102955818176, 0.27209386229515076, 0.05398734658956528, 0.26978081464767456, 0.29747360944747925, 0.41010212898254395, 0.0545368492603302, 0.2075808197259903, 0.040098268538713455, 0.20714770257472992, 0.3729572296142578, 0.3550529479980469, 0.4786156117916107, 0.11409778892993927, 0.27507272362709045, 0.1889083832502365, 0.2074296623468399, 0.4609362781047821, 0.20381633937358856, 0.14024963974952698, 0.037127669900655746, 0.05402032285928726, 0.47654542326927185], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
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
class TestPrimitiveOp_bb51c8dff7ad37662c159ad221bbdba5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.42281147837638855, 0.08266928791999817, 0.3753553628921509, 0.35161104798316956, 0.10693258047103882, 0.29845544695854187, 0.1096993014216423, 0.001929763937368989, 0.1978520154953003, 0.39333727955818176, 0.3993414044380188, 0.06251654773950577, 0.002844803035259247, 0.3152347207069397, 0.34154224395751953, 0.27489545941352844, 0.4788269102573395, 0.3868279755115509, 0.029443953186273575, 0.1249895766377449, 0.3196967542171478, 0.400125116109848, 0.4868909418582916, 0.15367543697357178], dtype='float32').reshape([24]),
            paddle.to_tensor([0.15911105275154114, 0.3191523849964142, 0.30547034740448, 0.19786109030246735, 0.3050064444541931, 0.30814671516418457, 0.1367986649274826, 0.30087926983833313, 0.015460798516869545, 0.16673791408538818, 0.33956626057624817, 0.444155216217041, 0.24342040717601776, 0.4141845405101776, 0.3069738447666168, 0.4979444146156311, 0.014999326318502426, 0.28377583622932434, 0.40204620361328125, 0.44927504658699036, 0.1680687516927719, 0.4651588797569275, 0.02933969907462597, 0.2827579081058502], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
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
class TestPrimitiveOp_fd89e9967f3b0dc3d74ac6b9521e9325(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3612076938152313, 0.39912208914756775, 0.14872072637081146, 0.25858673453330994, 0.07486964017152786, 0.33277249336242676, 0.43904757499694824, 0.2137400209903717, 0.2100633680820465, 0.4653855562210083, 0.47798794507980347, 0.301695853471756, 0.3093136250972748, 0.2589452266693115, 0.31297364830970764, 0.1711631417274475, 0.16097794473171234, 0.3179294168949127, 0.009308655746281147, 0.30526605248451233, 0.4128403663635254, 0.3444308936595917, 0.40020883083343506, 0.37081053853034973], dtype='float32').reshape([24]),
            paddle.to_tensor([0.38822075724601746, 0.06748425215482712, 0.28732869029045105, 0.2620127201080322, 0.2130596786737442, 0.4589809775352478, 0.17724978923797607, 0.19458502531051636, 0.12818197906017303, 0.2951882481575012, 0.04652812331914902, 0.3143450617790222, 0.120363250374794, 0.45104241371154785, 0.19639159739017487, 0.06870115548372269, 0.34110501408576965, 0.008029647171497345, 0.08939842879772186, 0.08021437376737595, 0.34604668617248535, 0.37177780270576477, 0.26150017976760864, 0.43491631746292114], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
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
class TestPrimitiveOp_e945b84a09d74a3c23aa419fbb33fb30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3243763744831085, 0.10922221839427948, 0.08571486920118332, 0.4509371221065521, 0.23479418456554413, 0.03712635114789009, 0.14491812884807587, 0.27291813492774963, 0.11994953453540802, 0.1481669545173645, 0.2928393483161926, 0.20127980411052704, 0.37972384691238403, 0.2028757780790329, 0.4817086458206177, 0.04140018671751022, 0.3429134786128998, 0.34433525800704956, 0.04517326503992081, 0.010226691141724586, 0.1278749257326126, 0.1376453936100006, 0.3240119218826294, 0.12344543635845184], dtype='float32').reshape([24]),
            paddle.to_tensor([0.4147098660469055, 0.15821808576583862, 0.145911306142807, 0.3174203634262085, 0.20298314094543457, 0.16132600605487823, 0.4538734257221222, 0.4587370455265045, 0.13601629436016083, 0.36636850237846375, 0.2857471704483032, 0.38322174549102783, 0.2860986888408661, 0.31123870611190796, 0.3196166157722473, 0.1634301245212555, 0.04482220858335495, 0.22947807610034943, 0.37294602394104004, 0.45078733563423157, 0.4548492729663849, 0.4942503869533539, 0.48787710070610046, 0.11250294744968414], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3ea672cb51f744b37f6b04871ad0bb9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3483818769454956, 0.3278091847896576, 0.41863152384757996, 0.13953231275081635, 0.34160345792770386, 0.4341847002506256, 0.10421984642744064, 0.10705380141735077, 0.3194532096385956, 0.23210681974887848, 0.4578392803668976, 0.01700410805642605, 0.17198330163955688, 0.31888848543167114, 0.3827630579471588, 0.2457941174507141, 0.05894274264574051, 0.0029358668252825737, 0.13500957190990448, 0.4977151155471802, 0.28242582082748413, 0.10478909313678741, 0.13626962900161743, 0.18728817999362946], dtype='float32').reshape([24]),
            paddle.to_tensor([0.4803254306316376, 0.43768441677093506, 0.207551971077919, 0.2566898465156555, 0.4935014545917511, 0.45660173892974854, 0.0014975254889577627, 0.267597496509552, 0.13524766266345978, 0.08755500614643097, 0.24130550026893616, 0.009236403740942478, 0.13304661214351654, 0.11607170104980469, 0.24687431752681732, 0.49902647733688354, 0.3803535997867584, 0.005215034820139408, 0.45766791701316833, 0.043826859444379807, 0.3714032471179962, 0.0043741073459386826, 0.02027713693678379, 0.09170043468475342], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
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
class TestPrimitiveOp_441aa607abb0bf2732a0a381ec81a55c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.40468478202819824, 0.20102308690547943, 0.21412764489650726, 0.34881195425987244, 0.15954531729221344, 0.3906097114086151, 0.41616368293762207, 0.2702464163303375, 0.2522966265678406, 0.19539105892181396, 0.34695497155189514, 0.49301624298095703, 0.4627770781517029, 0.037644173949956894, 0.4723157286643982, 0.36092641949653625, 0.2291240394115448, 0.21941101551055908, 0.3134632408618927, 0.39569878578186035, 0.04998055472970009, 0.3826083540916443, 0.31780657172203064, 0.08410269767045975], dtype='float32').reshape([24]),
            paddle.to_tensor([0.42166557908058167, 0.10020256787538528, 0.08045077323913574, 0.33110666275024414, 0.28873327374458313, 0.4704386591911316, 0.28217917680740356, 0.007594989612698555, 0.3561777174472809, 0.2565262019634247, 0.20871034264564514, 0.020144717767834663, 0.09739082306623459, 0.28244465589523315, 0.11376169323921204, 0.38699597120285034, 0.17603150010108948, 0.4838292598724365, 0.0001272702938877046, 0.1008094772696495, 0.1325010508298874, 0.3866124153137207, 0.13919435441493988, 0.395795077085495], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
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
class TestPrimitiveOp_6db8cfbb402bf539adb7b7c3a58785c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0210966058075428, 0.1405358612537384, 0.050669193267822266, 0.3962388336658478, 0.38744020462036133, 0.20083282887935638, 0.33778536319732666, 0.1588030755519867, 0.199144646525383, 0.33083635568618774, 0.18270771205425262, 0.46066367626190186, 0.17221054434776306, 0.12938076257705688, 0.32470399141311646, 0.2912845313549042, 0.3386044502258301, 0.24730175733566284, 0.08052926510572433, 0.2130158245563507, 0.1700984388589859, 0.26751574873924255, 0.37276071310043335, 0.44918128848075867], dtype='float32').reshape([24]),
            paddle.to_tensor([0.31895291805267334, 0.020774824544787407, 0.01710502803325653, 0.26318830251693726, 0.34087857604026794, 0.28989318013191223, 0.4662320911884308, 0.2854779064655304, 0.47583773732185364, 0.0061882007867097855, 0.30896711349487305, 0.007644944358617067, 0.018627822399139404, 0.45214036107063293, 0.05633212625980377, 0.0832122340798378, 0.3543674945831299, 0.29360178112983704, 0.3896822929382324, 0.018625479191541672, 0.4171549081802368, 0.46552833914756775, 0.08604802191257477, 0.21988141536712646], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_21953c011c5ec979f0ee471e5ded1ce5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.38297128677368164, 0.010262361727654934, 0.26849040389060974, 0.400585800409317, 0.023237742483615875, 0.33608707785606384, 0.49511146545410156, 0.42044952511787415, 0.14368371665477753, 0.2012222856283188, 0.4206011891365051, 0.02385561726987362, 0.0046110618859529495, 0.034425243735313416, 0.38097113370895386, 0.17182743549346924, 0.40061092376708984, 0.4605620801448822, 0.1295219212770462, 0.34910809993743896, 0.10261493176221848, 0.3077925145626068, 0.1498202383518219, 0.3907204568386078], dtype='float32').reshape([24]),
            paddle.to_tensor([0.10247986763715744, 0.4407726228237152, 0.37675759196281433, 0.11132825911045074, 0.009437788277864456, 0.011341910809278488, 0.3698680102825165, 0.016509748995304108, 0.17746783792972565, 0.1774677038192749, 0.01021084375679493, 0.3862353265285492, 0.0005669162492267787, 0.013267683796584606, 0.28642240166664124, 0.34295082092285156, 0.39364948868751526, 0.006558415479958057, 0.057813942432403564, 0.04779807850718498, 0.4497207999229431, 0.04904820770025253, 0.07234026491641998, 0.38010671734809875], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dd0025cd6a45187269c7168394faf3b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2933265268802643, 0.16921186447143555, 0.01854425109922886, 0.487466424703598, 0.20632091164588928, 0.46411484479904175, 0.3132723569869995, 0.4928017258644104, 0.19609762728214264, 0.10990948975086212, 0.18288911879062653, 0.35555291175842285, 0.14033372700214386, 0.0720382034778595, 0.4805426299571991, 0.21052047610282898, 0.26630765199661255, 0.3649366796016693, 0.29626673460006714, 0.22779420018196106, 0.19369280338287354, 0.4335307478904724, 0.43144491314888, 0.49565356969833374], dtype='float32').reshape([24]),
            paddle.to_tensor([0.31373128294944763, 0.4824369549751282, 0.25017398595809937, 0.2534303069114685, 0.42491424083709717, 0.2757776379585266, 0.44201698899269104, 0.2650067210197449, 0.33668312430381775, 0.4967504143714905, 0.4213058650493622, 0.3587561547756195, 0.40281763672828674, 0.37119513750076294, 0.19323843717575073, 0.09903126955032349, 0.03534949570894241, 0.36447274684906006, 0.05788661167025566, 0.3030148148536682, 0.04634292051196098, 0.49821725487709045, 0.2200348973274231, 0.01853361912071705], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6395898c00784190d9ebd969ca5e9851(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.015485916286706924, 0.3460428714752197, 0.41551101207733154, 0.06476900726556778, 0.17843209207057953, 0.4628934860229492, 0.1199483722448349, 0.26321372389793396, 0.2663916349411011, 0.11708617955446243, 0.047969717532396317, 0.05378718301653862, 0.27159857749938965, 0.47233375906944275, 0.4518211781978607, 0.37067002058029175, 0.4557918906211853, 0.045893386006355286, 0.4446979761123657, 0.40272200107574463, 0.25079548358917236, 0.28339874744415283, 0.1432686299085617, 0.44945579767227173], dtype='float32').reshape([24]),
            paddle.to_tensor([0.14335894584655762, 0.06016656011343002, 0.3126979470252991, 0.059239644557237625, 0.46179819107055664, 0.2539837956428528, 0.05213729664683342, 0.2744452953338623, 0.12494036555290222, 0.26780304312705994, 0.16777877509593964, 0.39798393845558167, 0.1627836525440216, 0.026539918035268784, 0.11397183686494827, 0.4219403862953186, 0.05174480751156807, 0.3475891649723053, 0.04770568758249283, 0.005230615381151438, 0.2112313061952591, 0.47769203782081604, 0.490946501493454, 0.11258085072040558], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
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
class TestPrimitiveOp_99d328083e86523bf4dfd07ca84c2793(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.09756629914045334, 0.09801867604255676, 0.30038389563560486, 0.45932257175445557, 0.07898318767547607, 0.007377479691058397, 0.20695184171199799, 0.33387717604637146, 0.032677263021469116, 0.18860885500907898, 0.021236848086118698, 0.03667052462697029, 0.41663211584091187, 0.3359481692314148, 0.25308653712272644, 0.32621538639068604, 0.13155707716941833, 0.4317314624786377, 0.22465412318706512, 0.35051223635673523, 0.3950556814670563, 0.2877831757068634, 0.04236484318971634, 0.34339913725852966], dtype='float32').reshape([24]),
            paddle.to_tensor([0.09727225452661514, 0.40842491388320923, 0.4322351813316345, 0.05528852716088295, 0.3019605875015259, 0.43542349338531494, 0.12367310374975204, 0.2120688408613205, 0.05228736996650696, 0.03723984584212303, 0.1830877959728241, 0.2512466311454773, 0.05638841539621353, 0.11866109073162079, 0.1470564901828766, 0.30375203490257263, 0.007476666010916233, 0.21952953934669495, 0.04817856848239899, 0.48219555616378784, 0.17738263309001923, 0.07973664999008179, 0.03273080289363861, 0.3500777781009674], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
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
class TestPrimitiveOp_14d308bc0644952a39a8066c30f1de95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.28283435106277466, 0.0676804631948471, 0.35826951265335083, 0.0053129130974411964, 0.23308177292346954, 0.05164271220564842, 0.39908045530319214, 0.43173283338546753, 0.45630308985710144, 0.40910929441452026, 0.4483509957790375, 0.14918763935565948, 0.2299223095178604, 0.40071001648902893, 0.25179004669189453, 0.49062275886535645, 0.3868352174758911, 0.1141994446516037, 0.2313208132982254, 0.45274618268013, 0.4197809398174286, 0.49996307492256165, 0.03991374745965004, 0.11095697432756424], dtype='float32').reshape([24]),
            paddle.to_tensor([0.15641798079013824, 0.2062762826681137, 0.3169999122619629, 0.11760962754487991, 0.4860123097896576, 0.41281992197036743, 0.07780268788337708, 0.3878086507320404, 0.09807993471622467, 0.19513367116451263, 0.3733014166355133, 0.34576311707496643, 0.4598023593425751, 0.08605511486530304, 0.3005611002445221, 0.3171677589416504, 0.11540449410676956, 0.453881174325943, 0.4378935992717743, 0.2092243731021881, 0.45806601643562317, 0.11833133548498154, 0.02251986414194107, 0.47287923097610474], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e937febf57beff439624249e8b4f286f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.22216951847076416, 0.4717188775539398, 0.4249148666858673, 0.27142882347106934, 0.1327742636203766, 0.28290072083473206, 0.4418167471885681, 0.11034416407346725, 0.4712463915348053, 0.49393007159233093, 0.3123990595340729, 0.1901472806930542, 0.08527329564094543, 0.3829624056816101, 0.29574015736579895, 0.011238908395171165, 0.284287691116333, 0.49747249484062195, 0.1427219808101654, 0.4186911880970001, 0.3912399709224701, 0.06630081683397293, 0.46670734882354736, 0.04763069748878479], dtype='float32').reshape([24]),
            paddle.to_tensor([0.0681866928935051, 0.16083848476409912, 0.22945405542850494, 0.35928285121917725, 0.12687727808952332, 0.4214862883090973, 0.42756494879722595, 0.27913698554039, 0.16280722618103027, 0.2662515342235565, 0.08638124912977219, 0.30440986156463623, 0.004144219681620598, 0.019412312656641006, 0.3725771903991699, 0.05502048879861832, 0.022154243662953377, 0.030937639996409416, 0.13815757632255554, 0.09306534379720688, 0.02362733706831932, 0.24921591579914093, 0.398305743932724, 0.20674848556518555], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
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
class TestPrimitiveOp_9204d5657e0c0e62f15816bd8ce56e5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3460783362388611, 0.03443462774157524, 0.408433735370636, 0.48116007447242737, 0.16743560135364532, 0.085225909948349, 0.4584450125694275, 0.006579235196113586, 0.08808024972677231, 0.22657927870750427, 0.29153281450271606, 0.49339964985847473, 0.29302096366882324, 0.3056832253932953, 0.13836996257305145, 0.40149006247520447, 0.3809690773487091, 0.011306263506412506, 0.15014468133449554, 0.3011144995689392, 0.4817412197589874, 0.2781399190425873, 0.3353548049926758, 0.11786077916622162], dtype='float32').reshape([24]),
            paddle.to_tensor([0.15490134060382843, 0.3359456956386566, 0.1523849368095398, 0.45645374059677124, 0.41901591420173645, 0.23732852935791016, 0.12281060218811035, 0.38296806812286377, 0.06538064032793045, 0.30907142162323, 0.4211535155773163, 0.18149176239967346, 0.4729141592979431, 0.2542508840560913, 0.3233759105205536, 0.2626575827598572, 0.414181649684906, 0.22685237228870392, 0.017079800367355347, 0.2565867006778717, 0.18213969469070435, 0.15655970573425293, 0.4355728328227997, 0.16980616748332977], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
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
class TestPrimitiveOp_b2c2c9751ce2a29b5ee3a07eb9773142(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1823611706495285, 0.3962276875972748, 0.4280262887477875, 0.12127076089382172, 0.11854170262813568, 0.43120452761650085, 0.31985846161842346, 0.41223856806755066, 0.3951638340950012, 0.4401490390300751, 0.4098871350288391, 0.12045428901910782, 0.36951667070388794, 0.3026745915412903, 0.18505096435546875, 0.007786941714584827, 0.24096012115478516, 0.12980540096759796, 0.49633386731147766, 0.3096555769443512, 0.2751445472240448, 0.18698763847351074, 0.4963749349117279, 0.4187670052051544], dtype='float32').reshape([24]),
            paddle.to_tensor([0.2842351198196411, 0.029225170612335205, 0.03564634174108505, 0.26090165972709656, 0.22109831869602203, 0.4564916789531708, 0.053904809057712555, 0.1938057243824005, 0.29531407356262207, 0.3922020494937897, 0.24451473355293274, 0.39571478962898254, 0.28309890627861023, 0.28385430574417114, 0.4339904189109802, 0.34302595257759094, 0.009946681559085846, 0.268003910779953, 0.0916522666811943, 0.3937080204486847, 0.4599798023700714, 0.009313804097473621, 0.4707798659801483, 0.45024019479751587], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cd944cbc829b4f633b87a5eb3ba46711(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3191579580307007, 0.2731638550758362, 0.3224371075630188, 0.031068051233887672, 0.1186930239200592, 0.3118444085121155, 0.40523266792297363, 0.40708720684051514, 0.10220556706190109, 0.41578182578086853, 0.47054848074913025, 0.3298671543598175, 0.23831669986248016, 0.18965032696723938, 0.4976508319377899, 0.23149944841861725, 0.2474496215581894, 0.2674003839492798, 0.17953433096408844, 0.4432717263698578, 0.35693398118019104, 0.4385075867176056, 0.32127657532691956, 0.460958868265152], dtype='float32').reshape([24]),
            paddle.to_tensor([0.007759390398859978, 0.39703166484832764, 0.14795221388339996, 0.20501507818698883, 0.07570552080869675, 0.33046460151672363, 0.053221795707941055, 0.1448243260383606, 0.39755159616470337, 0.1425670087337494, 0.020017744973301888, 0.2278515249490738, 0.03960034251213074, 0.21489393711090088, 0.1418943703174591, 0.4585473835468292, 0.40421321988105774, 0.12636959552764893, 0.315336674451828, 0.25199800729751587, 0.4459870457649231, 0.1506226360797882, 0.12882934510707855, 0.25234848260879517], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_042ec3c2cdc34f0deae9ac33e7557443(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.27327877283096313, 0.3870028853416443, 0.06256396323442459, 0.02868662402033806, 0.3612062633037567, 0.12573139369487762, 0.011210931465029716, 0.36385631561279297, 0.008875143714249134, 0.16479448974132538, 0.4837847054004669, 0.20957621932029724, 0.1223117932677269, 0.09353309124708176, 0.2946908175945282, 0.40447285771369934, 0.25413790345191956, 0.07305101305246353, 0.4619729220867157, 0.09364607185125351, 0.08762020617723465, 0.2246808409690857, 0.4245934784412384, 0.018741071224212646], dtype='float32').reshape([24]),
            paddle.to_tensor([0.16428643465042114, 0.008446759544312954, 0.14582963287830353, 0.13918288052082062, 0.26555201411247253, 0.2110971063375473, 0.4846075773239136, 0.050534699112176895, 0.1081136092543602, 0.2533041834831238, 0.16036318242549896, 0.2517251968383789, 0.4891287088394165, 0.10519136488437653, 0.491793692111969, 0.3665371239185333, 0.321613073348999, 0.3481799066066742, 0.3071436882019043, 0.4702399969100952, 0.0004998861695639789, 0.044105228036642075, 0.440019428730011, 0.4786377251148224], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
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
class TestPrimitiveOp_f1a21c44814d508bb39431b14960aebb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.35693100094795227, 0.49161261320114136, 0.3728858232498169, 0.06533476710319519, 0.31946367025375366, 0.22106540203094482, 0.04963909462094307, 0.46916940808296204, 0.2829035222530365, 0.06328136473894119, 0.13949783146381378, 0.060443803668022156, 0.23511122167110443, 0.4260987341403961, 0.47255101799964905, 0.12235407531261444, 0.4374164342880249, 0.24069102108478546, 0.3153023421764374, 0.1330728679895401, 0.0725119411945343, 0.21814101934432983, 0.14752711355686188, 0.12983427941799164], dtype='float32').reshape([24]),
            paddle.to_tensor([0.022101860493421555, 0.3643698990345001, 0.14877784252166748, 0.008698607794940472, 0.20557324588298798, 0.27467259764671326, 0.04583905637264252, 0.08336306363344193, 0.40658047795295715, 0.2563936710357666, 0.4374331831932068, 0.1845434606075287, 0.3933418095111847, 0.16926386952400208, 0.4192522168159485, 0.4040462374687195, 0.049375273287296295, 0.04602387920022011, 0.3717819154262543, 0.02422940731048584, 0.49567240476608276, 0.33613118529319763, 0.365788072347641, 0.136431023478508], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
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
class TestPrimitiveOp_92730371ccf92b7a7e3e36c3c2b1895b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.06304154545068741, 0.29864418506622314, 0.1388540416955948, 0.39037221670150757, 0.30445072054862976, 0.13776838779449463, 0.43298131227493286, 0.4721490144729614, 0.17988599836826324, 0.46066993474960327, 0.24454152584075928, 0.3099138140678406, 0.018890922889113426, 0.16745896637439728, 0.023814519867300987, 0.29015523195266724, 0.4948238134384155, 0.442834734916687, 0.43423059582710266, 0.22957034409046173, 0.4154045283794403, 0.01994956284761429, 0.4893243610858917, 0.21568860113620758], dtype='float32').reshape([24]),
            paddle.to_tensor([0.06159462779760361, 0.17987963557243347, 0.2939353585243225, 0.40622374415397644, 0.016140853986144066, 0.010005838237702847, 0.3262770473957062, 0.04786226153373718, 0.006287831813097, 0.1692119538784027, 0.439468115568161, 0.35572150349617004, 0.45436128973960876, 0.3452468812465668, 0.4358096420764923, 0.358969122171402, 0.30024629831314087, 0.08428844809532166, 0.4258826673030853, 0.24552342295646667, 0.21239078044891357, 0.4502423405647278, 0.4897916316986084, 0.24511180818080902], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_90c56de870b531a6e7ea1b7f9af3da1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2686774730682373, 0.015429876744747162, 0.008320430293679237, 0.2537198066711426, 0.37676122784614563, 0.3731080889701843, 0.05977616831660271, 0.0696033164858818, 0.4978320598602295, 0.3556686043739319, 0.46474823355674744, 0.18952178955078125, 0.3007684648036957, 0.1707219034433365, 0.4185183048248291, 0.15997563302516937, 0.1978771984577179, 0.09467022120952606, 0.35740023851394653, 0.15861745178699493, 0.39399126172065735, 0.44170236587524414, 0.1843089908361435, 0.1804286390542984], dtype='float32').reshape([24]),
            paddle.to_tensor([0.4174814820289612, 0.35215041041374207, 0.34631291031837463, 0.02836156263947487, 0.4176118075847626, 0.4666920006275177, 0.31074315309524536, 0.45024096965789795, 0.30359527468681335, 0.07473526895046234, 0.01515351515263319, 0.46556949615478516, 0.15265941619873047, 0.22525517642498016, 0.008867619559168816, 0.11759386956691742, 0.34550464153289795, 0.21290560066699982, 0.4914778470993042, 0.2836286425590515, 0.057972781360149384, 0.27428141236305237, 0.26461225748062134, 0.05772741138935089], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
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
class TestPrimitiveOp_eb57e6727310bac3d5d51fee83b272a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4881134033203125, 0.11468533426523209, 0.2612071931362152, 0.38030245900154114, 0.021396568045020103, 0.45309773087501526, 0.16356751322746277, 0.11987242102622986, 0.11192021518945694, 0.027293849736452103, 0.47223973274230957, 0.47053709626197815, 0.0325610525906086, 0.06356102228164673, 0.3113722801208496, 0.0007756504346616566, 0.08912140876054764, 0.4233382046222687, 0.03237156569957733, 0.36110302805900574, 0.022093860432505608, 0.1379624754190445, 0.27979791164398193, 0.4354380667209625], dtype='float32').reshape([24]),
            paddle.to_tensor([0.1689424365758896, 0.30405688285827637, 0.38950446248054504, 0.16508503258228302, 0.3236425518989563, 0.2509315311908722, 0.15311509370803833, 0.40804144740104675, 0.42480573058128357, 0.024796435609459877, 0.14060787856578827, 0.002965668449178338, 0.42817965149879456, 0.4381483793258667, 0.2640034258365631, 0.23062065243721008, 0.03668816760182381, 0.3544042110443115, 0.30383992195129395, 0.16835841536521912, 0.2993701696395874, 0.004579231142997742, 0.40677696466445923, 0.3733751177787781], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
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
class TestPrimitiveOp_bc3843c21888c71d1c32c41c0ebb96d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.12513338029384613, 0.18403656780719757, 0.27308550477027893, 0.42414334416389465, 0.17093493044376373, 0.14989915490150452, 0.37266695499420166, 0.28934377431869507, 0.00896511971950531, 0.1469772905111313, 0.3187383711338043, 0.05427708849310875, 0.294699490070343, 0.20837841928005219, 0.49950119853019714, 0.060025960206985474, 0.376433789730072, 0.0905025452375412, 0.1500069797039032, 0.12956218421459198, 0.3482898771762848, 0.34389713406562805, 0.46167463064193726, 0.42420631647109985], dtype='float32').reshape([24]),
            paddle.to_tensor([0.39863619208335876, 0.37510836124420166, 0.01863253489136696, 0.40454208850860596, 0.3168427646160126, 0.33005544543266296, 0.22674895823001862, 0.33450809121131897, 0.4460521638393402, 0.3933388292789459, 0.4529915153980255, 0.23362481594085693, 0.07341182231903076, 0.38512852787971497, 0.38534075021743774, 0.35783013701438904, 0.4853615164756775, 0.4360749125480652, 0.16441334784030914, 0.10529271513223648, 0.31381669640541077, 0.40715497732162476, 0.3519487679004669, 0.08888618648052216], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
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
class TestPrimitiveOp_5ea066aee2abc040f4dfc0ad980b11e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.07127968221902847, 0.09125033766031265, 0.023202193900942802, 0.3497844934463501, 0.3141830563545227, 0.2902604937553406, 0.15621709823608398, 0.1674126237630844, 0.16408847272396088, 0.24759742617607117, 0.042430564761161804, 0.10406863689422607, 0.4620820879936218, 0.020261798053979874, 0.1034533753991127, 0.3168523907661438, 0.0856364369392395, 0.38661009073257446, 0.30293703079223633, 0.4134373366832733, 0.005359933711588383, 0.21869277954101562, 0.3607705533504486, 0.34248507022857666], dtype='float32').reshape([24]),
            paddle.to_tensor([0.09426507353782654, 0.3384673297405243, 0.2706443667411804, 0.3504253923892975, 0.1125011295080185, 0.1436195820569992, 0.05829785764217377, 0.0059114666655659676, 0.1310701221227646, 0.21837809681892395, 0.19304171204566956, 0.14124375581741333, 0.021088536828756332, 0.256740003824234, 0.01590237393975258, 0.3753992021083832, 0.3164536952972412, 0.05302175134420395, 0.09527934342622757, 0.406658411026001, 0.3974762260913849, 0.2860569357872009, 0.491249680519104, 0.11801835894584656], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a177e7d6c20dd12290803fb79f8a5c1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.40648627281188965, 0.24538792669773102, 0.48398125171661377, 0.42668208479881287, 0.22706156969070435, 0.21326547861099243, 0.04420938715338707, 0.4704393744468689, 0.23162122070789337, 0.2600278854370117, 0.23639260232448578, 0.09334257245063782, 0.43995553255081177, 0.34260687232017517, 0.04798019304871559, 0.3543713688850403, 0.15904739499092102, 0.13867199420928955, 0.20997390151023865, 0.15397725999355316, 0.1087106391787529, 0.2716803550720215, 0.1969616860151291, 0.2087760865688324], dtype='float32').reshape([24]),
            paddle.to_tensor([0.30893054604530334, 0.20366156101226807, 0.42002734541893005, 0.48796215653419495, 0.029658503830432892, 0.0772138461470604, 0.0037884870544075966, 0.24181978404521942, 0.28350767493247986, 0.28913992643356323, 0.4851824641227722, 0.18901866674423218, 0.0048148250207304955, 0.4520302414894104, 0.1805366426706314, 0.3485776484012604, 0.013709526509046555, 0.36075249314308167, 0.057362738996744156, 0.03830808401107788, 0.1693912297487259, 0.27795732021331787, 0.04746037721633911, 0.28284844756126404], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6af2d43ac434a7ac953c165c68f733b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4089096188545227, 0.1885455697774887, 0.06229705363512039, 0.025573620572686195, 0.40406835079193115, 0.3970227539539337, 0.31646642088890076, 0.4974219799041748, 0.48800355195999146, 0.19977805018424988, 0.4191742539405823, 0.4992269277572632, 0.1220344752073288, 0.3901118338108063, 0.47217321395874023, 0.3088521957397461, 0.04744509980082512, 0.19158270955085754, 0.4859465956687927, 0.00586653221398592, 0.011697019450366497, 0.03876720741391182, 0.314231812953949, 0.49260035157203674], dtype='float32').reshape([24]),
            paddle.to_tensor([0.29054465889930725, 0.20007146894931793, 0.0699198991060257, 0.08059374988079071, 0.46132805943489075, 0.3376687467098236, 0.4251303970813751, 0.4897180199623108, 0.22213463485240936, 0.0775279775261879, 0.4908750653266907, 0.3188328742980957, 0.3673385679721832, 0.29833224415779114, 0.4389877915382385, 0.2682447135448456, 0.396658718585968, 0.15508116781711578, 0.0841103047132492, 0.10179045051336288, 0.10187768191099167, 0.48471173644065857, 0.21435533463954926, 0.06461380422115326], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7a0c65c878c6cddce81bb5164294bfc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4217832386493683, 0.26176580786705017, 0.42301103472709656, 0.06899531185626984, 0.49602118134498596, 0.23967638611793518, 0.17430457472801208, 0.4637378752231598, 0.13974443078041077, 0.07450118660926819, 0.011908818036317825, 0.49217328429222107, 0.30321693420410156, 0.3209482729434967, 0.2737846374511719, 0.4105992317199707, 0.4389213025569916, 0.3283942937850952, 0.02113870345056057, 0.05370422452688217, 0.04829084128141403, 0.494429349899292, 0.3366137742996216, 0.07904564589262009], dtype='float32').reshape([24]),
            paddle.to_tensor([0.10020922124385834, 0.36040979623794556, 0.3451809585094452, 0.08937521278858185, 0.04954676330089569, 0.2155519723892212, 0.06478944420814514, 0.019300175830721855, 0.2674526870250702, 0.4500877261161804, 0.2251822054386139, 0.2585299015045166, 0.10121611505746841, 0.40257906913757324, 0.30198734998703003, 0.30425578355789185, 0.37847697734832764, 0.3676827847957611, 0.09500036388635635, 0.2978985011577606, 0.3220682144165039, 0.31413328647613525, 0.20751874148845673, 0.12527833878993988], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f7c79cf730452c0bc1e643bbd8536b40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.41577062010765076, 0.4890482425689697, 0.03013896569609642, 0.3531072437763214, 0.3160388171672821, 0.24565263092517853, 0.07122120261192322, 0.3650830388069153, 0.2216411679983139, 0.23102734982967377, 0.09765178710222244, 0.4567965567111969, 0.1789235770702362, 0.3257059156894684, 0.0029165558516979218, 0.3824552595615387, 0.11337274312973022, 0.4016767740249634, 0.4744713604450226, 0.4593200087547302, 0.16416361927986145, 0.13903988897800446, 0.47507229447364807, 0.37058204412460327], dtype='float32').reshape([24]),
            paddle.to_tensor([0.03755776584148407, 0.21775440871715546, 0.23867663741111755, 0.3571220338344574, 0.0036783162504434586, 0.3568361699581146, 0.3765510320663452, 0.39102163910865784, 0.44721299409866333, 0.012020466849207878, 0.3160547912120819, 0.029246484860777855, 0.27084386348724365, 0.11951540410518646, 0.041485194116830826, 0.4464377164840698, 0.33930811285972595, 0.4862894117832184, 0.24695613980293274, 0.27994653582572937, 0.03206433728337288, 0.3237987160682678, 0.2470344603061676, 0.34010496735572815], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
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
class TestPrimitiveOp_8e3f5a0c6d9e588cb4a361d11b32c10d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.24747993052005768, 0.08444679528474808, 0.4609930217266083, 0.3207951486110687, 0.4351491332054138, 0.3688061535358429, 0.2839893102645874, 0.3300152122974396, 0.41294193267822266, 0.3649066090583801, 0.3473729193210602, 0.4198988378047943, 0.1625453680753708, 0.08249597996473312, 0.38573628664016724, 0.294180303812027, 0.19753725826740265, 0.08508450537919998, 0.4258096218109131, 0.2672591209411621, 0.39229899644851685, 0.4606698751449585, 0.2101658284664154, 0.4031069576740265], dtype='float32').reshape([24]),
            paddle.to_tensor([0.0938786193728447, 0.30581432580947876, 0.01788550801575184, 0.05210574343800545, 0.31671133637428284, 0.10273763537406921, 0.2759387195110321, 0.4677678346633911, 0.2512931525707245, 0.18504437804222107, 0.06472693383693695, 0.43672898411750793, 0.38604217767715454, 0.07011484354734421, 0.4461390972137451, 0.49022847414016724, 0.06516890972852707, 0.4210980236530304, 0.428902268409729, 0.08478165417909622, 0.22496168315410614, 0.34644636511802673, 0.078738734126091, 0.30314740538597107], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
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
class TestPrimitiveOp_859916938deec8970a205303c16975ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4347810447216034, 0.15940269827842712, 0.17664343118667603, 0.3824494481086731, 0.24352580308914185, 0.33991098403930664, 0.42089536786079407, 0.04446713998913765, 0.3208412230014801, 0.33512669801712036, 0.21370652318000793, 0.04180944338440895, 0.03875293955206871, 0.17902922630310059, 0.09477446973323822, 0.20786434412002563, 0.22030729055404663, 0.462057888507843, 0.288800984621048, 0.21179451048374176, 0.030562235042452812, 0.4673629701137543, 0.3987131416797638, 0.151397705078125], dtype='float32').reshape([24]),
            paddle.to_tensor([0.12115351110696793, 0.08800000697374344, 0.20509029924869537, 0.4517628848552704, 0.41291823983192444, 0.14662465453147888, 0.3803623616695404, 0.10470513254404068, 0.4940492808818817, 0.20152774453163147, 0.4801822304725647, 0.4221990704536438, 0.2589115798473358, 0.14514170587062836, 0.3002837896347046, 0.03473708778619766, 0.10798722505569458, 0.4809415340423584, 0.22394360601902008, 0.37412530183792114, 0.04837535694241524, 0.493552565574646, 0.35049429535865784, 0.0923210084438324], dtype='float32').reshape([24]),
        ]


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
class TestPrimitiveOp_5b61d8b3e5273ba9b1cbcbf95a3746ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.18557792901992798, 0.32007303833961487, 0.18999125063419342, 0.07081961631774902, 0.382150799036026, 0.3697015345096588, 0.34371039271354675, 0.2460322380065918, 0.4014826714992523, 0.13120827078819275, 0.21110233664512634, 0.03582832217216492, 0.11002826690673828, 0.48579031229019165, 0.47055140137672424, 0.49326616525650024, 0.02975228615105152, 0.025987448170781136, 0.017184214666485786, 0.025708984583616257, 0.032728683203458786, 0.2615104913711548, 0.32695814967155457, 0.13774420320987701], dtype='float32').reshape([24]),
            paddle.to_tensor([0.12575137615203857, 0.4861892759799957, 0.15025468170642853, 0.35640373826026917, 0.47453585267066956, 0.33910202980041504, 0.4794483482837677, 0.1600324511528015, 0.11602745205163956, 0.008278917521238327, 0.02693174220621586, 0.4506690502166748, 0.2366214245557785, 0.011804630048573017, 0.18478918075561523, 0.27150222659111023, 0.46170493960380554, 0.03931446373462677, 0.1282913237810135, 0.42320048809051514, 0.2102435827255249, 0.2237933874130249, 0.15707646310329437, 0.392195463180542], dtype='float32').reshape([24]),
        ]


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
class TestPrimitiveOp_d34976883d7ec4351ab469e5bccb7b7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4894232749938965, 0.39302146434783936, 0.24774377048015594, 0.3282211124897003, 0.07065164297819138, 0.44173985719680786, 0.2776108682155609, 0.3275449275970459, 0.21766872704029083, 0.4223199784755707, 0.21058465540409088, 0.25269728899002075, 0.23878751695156097, 0.46747153997421265, 0.25833195447921753, 0.22710029780864716, 0.3212806284427643, 0.4295365810394287, 0.008130498230457306, 0.4297127425670624, 0.3733919858932495, 0.014107746072113514, 0.21050621569156647, 0.010991958901286125], dtype='float32').reshape([24]),
            paddle.to_tensor([0.4324173927307129, 0.22862039506435394, 0.49999871850013733, 0.3409220576286316, 0.11399386823177338, 0.39413684606552124, 0.0726449266076088, 0.31914374232292175, 0.35738691687583923, 0.4450126886367798, 0.3715563714504242, 0.08431700617074966, 0.2281743437051773, 0.12527106702327728, 0.0037365176249295473, 0.40489575266838074, 0.43690553307533264, 0.306119441986084, 0.17036275565624237, 0.01178570743650198, 0.07605984061956406, 0.41606131196022034, 0.2228042334318161, 0.020824164152145386], dtype='float32').reshape([24]),
        ]


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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c1b79f8ef8fb6e10cc7ef0d99c17ef5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.03513530269265175, 0.3028586208820343, 0.2280714362859726, 0.37589627504348755, 0.4481565058231354, 0.36293357610702515, 0.18754443526268005, 0.1812683641910553, 0.20437921583652496, 0.45626455545425415, 0.3015091121196747, 0.2065226286649704, 0.38265523314476013, 0.07587423175573349, 0.0011120333801954985, 0.046214550733566284, 0.31792333722114563, 0.24716724455356598, 0.06855440884828568, 0.04069637879729271, 0.41614317893981934, 0.12556323409080505, 0.10468371212482452, 0.3293677270412445], dtype='float32').reshape([24]),
            paddle.to_tensor([0.055061038583517075, 0.3164403736591339, 0.2714866101741791, 0.258524626493454, 0.3732859790325165, 0.36024224758148193, 0.045409683138132095, 0.39449068903923035, 0.2997606098651886, 0.26498129963874817, 0.4412654936313629, 0.2794581651687622, 0.13612616062164307, 0.42093876004219055, 0.38086020946502686, 0.42723509669303894, 0.048407308757305145, 0.25255683064460754, 0.10977859050035477, 0.16724924743175507, 0.19889523088932037, 0.20510996878147125, 0.45520102977752686, 0.3015405833721161], dtype='float32').reshape([24]),
        ]


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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c4cca45ba085083cdd5e6c18615afe24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0009266676497645676, 0.4569169580936432, 0.1922357976436615, 0.026860874146223068, 0.012805046513676643, 0.4691333472728729, 0.11356056481599808, 0.05572505295276642, 0.31730183959007263, 0.15613949298858643, 0.040136151015758514, 0.31188076734542847, 0.46710696816444397, 0.0707920491695404, 0.17637360095977783, 0.4591725766658783, 0.03263413906097412, 0.05042111128568649, 0.0019172467291355133, 0.14436832070350647, 0.373232901096344, 0.11397235095500946, 0.05069105699658394, 0.16660988330841064], dtype='float32').reshape([24]),
            paddle.to_tensor([0.360390841960907, 0.12386021018028259, 0.3096921145915985, 0.24416986107826233, 0.23045581579208374, 0.21057748794555664, 0.03576572239398956, 0.4865342676639557, 0.42694875597953796, 0.36033546924591064, 0.02399381250143051, 0.4239684045314789, 0.30213919281959534, 0.3426682651042938, 0.1406739503145218, 0.08832024037837982, 0.33539873361587524, 0.2701224088668823, 0.19719015061855316, 0.3361166715621948, 0.09247983247041702, 0.19217103719711304, 0.351728618144989, 0.2429441511631012], dtype='float32').reshape([24]),
        ]


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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ec1a9682767f6cb522640e180117e20b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2929288446903229, 0.40260937809944153, 0.46827730536460876, 0.2568630278110504, 0.21958377957344055, 0.39780861139297485, 0.09189960360527039, 0.3860931396484375, 0.006806256249547005, 0.30961287021636963, 0.4472181499004364, 0.009248164482414722, 0.30434098839759827, 0.019307266920804977, 0.1883336305618286, 0.44343698024749756, 0.1482214480638504, 0.10681938380002975, 0.24254009127616882, 0.44587817788124084, 0.21449114382266998, 0.13496658205986023, 0.38958510756492615, 0.3233550786972046], dtype='float32').reshape([24]),
            paddle.to_tensor([0.1038304939866066, 0.47783198952674866, 0.10749951750040054, 0.44453269243240356, 0.17322668433189392, 0.3538517951965332, 0.0059220134280622005, 0.3375278413295746, 0.3292349576950073, 0.16054484248161316, 0.2742246687412262, 0.09881531447172165, 0.425597608089447, 0.11965606361627579, 0.21171344816684723, 0.13014858961105347, 0.37691208720207214, 0.49213653802871704, 0.46061280369758606, 0.22359324991703033, 0.19646857678890228, 0.0014152233488857746, 0.29803889989852905, 0.47569239139556885], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1c1d90d6a3483c6fa7fe751a19ccea59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.46142858266830444, 0.470750629901886, 0.27092957496643066, 0.1454208493232727, 0.15598635375499725, 0.038589317351579666, 0.16466012597084045, 0.2982492446899414, 0.3323836326599121, 0.16400736570358276, 0.1523159146308899, 0.26609864830970764, 0.1981530487537384, 0.48086389899253845, 0.14776700735092163, 0.29769015312194824, 0.3693331778049469, 0.19759811460971832, 0.45496347546577454, 0.17112967371940613, 0.05407559499144554, 0.05871938541531563, 0.05141723155975342, 0.04344716668128967], dtype='float32').reshape([24]),
            paddle.to_tensor([0.3204044997692108, 0.1301552653312683, 0.4498136043548584, 0.07472959905862808, 0.30138176679611206, 0.031225264072418213, 0.3523406386375427, 0.3556104302406311, 0.2625509798526764, 0.08319670706987381, 0.44868937134742737, 0.47765398025512695, 0.2014644294977188, 0.4118974804878235, 0.03678882494568825, 0.042427487671375275, 0.1265745460987091, 0.15858441591262817, 0.47994405031204224, 0.3295329213142395, 0.00915518868714571, 0.37874555587768555, 0.3858391344547272, 0.2634248435497284], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ace3e8bb67a044498db915c35d6afe85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3852593004703522, 0.061754368245601654, 0.10513415932655334, 0.09169141948223114, 0.4158417582511902, 0.273942232131958, 0.07613906264305115, 0.12092604488134384, 0.42380741238594055, 0.22722835838794708, 0.49910327792167664, 0.2510910928249359, 0.28568777441978455, 0.3285799026489258, 0.1428675353527069, 0.3996749818325043, 0.37039443850517273, 0.1949162632226944, 0.05567727982997894, 0.19043688476085663, 0.4979543387889862, 0.02417675592005253, 0.4568227529525757, 0.3067842721939087], dtype='float32').reshape([24]),
            paddle.to_tensor([0.11194608360528946, 0.39028507471084595, 0.45009341835975647, 0.4008389115333557, 0.12800481915473938, 0.19818904995918274, 0.17486077547073364, 0.4536150395870209, 0.48820021748542786, 0.4709565341472626, 0.3319510221481323, 0.016767073422670364, 0.17767038941383362, 0.04683232307434082, 0.1732078343629837, 0.32106587290763855, 0.37855473160743713, 0.23165100812911987, 0.154708594083786, 0.3543301522731781, 0.22153975069522858, 0.25542330741882324, 0.1846991330385208, 0.45740818977355957], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a03862dd4c3065a3a2446454e91f4424(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.31615838408470154, 0.3888537585735321, 0.12670762836933136, 0.2716168165206909, 0.2606115937232971, 0.12328989803791046, 0.030999915674328804, 0.42016199231147766, 0.28645840287208557, 0.38627946376800537, 0.3130512535572052, 0.07873406261205673, 0.13075114786624908, 0.45866015553474426, 0.14900392293930054, 0.36201339960098267, 0.1644120216369629, 0.2197045534849167, 0.0004509511054493487, 0.2886301875114441, 0.3511703908443451, 0.2191217839717865, 0.3984113335609436, 0.39675962924957275], dtype='float32').reshape([24]),
            paddle.to_tensor([0.10704578459262848, 0.18470416963100433, 0.3558475971221924, 0.2785373330116272, 0.06888729333877563, 0.03866197168827057, 0.024407977238297462, 0.09313499927520752, 0.296229749917984, 0.0155866127461195, 0.33309775590896606, 0.04949507862329483, 0.31130826473236084, 0.03375934436917305, 0.4344682991504669, 0.41972437500953674, 0.23060400784015656, 0.20581530034542084, 0.03405694290995598, 0.3890611231327057, 0.020368656143546104, 0.1023167073726654, 0.47946467995643616, 0.06333756446838379], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_58ee8c07619df0f11ed78be82f8a13aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.030688457190990448, 0.2550361752510071, 0.10495477169752121, 0.40645548701286316, 0.46929818391799927, 0.17530560493469238, 0.3389422297477722, 0.16908644139766693, 0.1982177346944809, 0.4975198209285736, 0.14448674023151398, 0.28588640689849854, 0.2764745354652405, 0.04856292903423309, 0.2095988243818283, 0.08120176941156387, 0.03200673311948776, 0.2844375967979431, 0.1565363109111786, 0.02407647669315338, 0.4372400641441345, 0.3700490891933441, 0.11595606058835983, 0.0496029257774353], dtype='float32').reshape([24]),
            paddle.to_tensor([0.11437821388244629, 0.17038650810718536, 0.48310598731040955, 0.20973831415176392, 0.3210460841655731, 0.4348059296607971, 0.2197347730398178, 0.23732231557369232, 0.11183145642280579, 0.25240206718444824, 0.45429596304893494, 0.3402089476585388, 0.3089520037174225, 0.011120880953967571, 0.13550935685634613, 0.32128268480300903, 0.39819738268852234, 0.3210846781730652, 0.21410638093948364, 0.4513459801673889, 0.49914267659187317, 0.3547615706920624, 0.19369006156921387, 0.26052728295326233], dtype='float32').reshape([24]),
        ]


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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c9de7a593d19cd529be76bf11024604f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.03507506474852562, 0.4726286232471466, 0.21683938801288605, 0.47727131843566895, 0.2138596922159195, 0.15568046271800995, 0.3742484152317047, 0.051243461668491364, 0.163829505443573, 0.4451495110988617, 0.14013652503490448, 0.4116010069847107, 0.12977293133735657, 0.2837769389152527, 0.1449921876192093, 0.4418068826198578, 0.19956544041633606, 0.07563529163599014, 0.30375921726226807, 0.06277606636285782, 0.4096645712852478, 0.06267563998699188, 0.39228057861328125, 0.1536305695772171], dtype='float32').reshape([24]),
            paddle.to_tensor([0.23923535645008087, 0.03189670294523239, 0.31410402059555054, 0.35864272713661194, 0.22936499118804932, 0.2988196611404419, 0.4852163791656494, 0.1667562872171402, 0.19574952125549316, 0.07073426991701126, 0.08981485664844513, 0.24961453676223755, 0.03717724233865738, 0.40135490894317627, 0.37110987305641174, 0.25225988030433655, 0.4806106984615326, 0.43387702107429504, 0.22379320859909058, 0.02887541987001896, 0.4826131761074066, 0.4258623421192169, 0.3591468930244446, 0.35681867599487305], dtype='float32').reshape([24]),
        ]


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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_52c601e57d921e60e73cc0d4a79d6207(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3134850561618805, 0.3172972798347473, 0.38479962944984436, 0.3205386698246002, 0.44504889845848083, 0.3865216374397278, 0.22452259063720703, 0.15682221949100494, 0.30840229988098145, 0.07892552018165588, 0.4569323658943176, 0.11894088983535767, 0.46890756487846375, 0.06404297798871994, 0.4316891133785248, 0.33794549107551575, 0.32232537865638733, 0.17606180906295776, 0.21438266336917877, 0.22911998629570007, 0.4944925308227539, 0.4872172474861145, 0.3546540141105652, 0.3062010407447815], dtype='float32').reshape([24]),
            paddle.to_tensor([0.23078526556491852, 0.3358755111694336, 0.14085254073143005, 0.011944515630602837, 0.29519596695899963, 0.15833447873592377, 0.05954698100686073, 0.4254775643348694, 0.46309539675712585, 0.199662983417511, 0.47849521040916443, 0.2883360981941223, 0.4451708495616913, 0.2812700569629669, 0.010555868037045002, 0.15761737525463104, 0.015374504961073399, 0.04203938692808151, 0.3856744170188904, 0.49273914098739624, 0.07863856106996536, 0.43883219361305237, 0.4896291494369507, 0.34693238139152527], dtype='float32').reshape([24]),
        ]


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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8beb713ddd2cd5a19314b01e62f842e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1759195774793625, 0.13655953109264374, 0.08970408141613007, 0.4518701434135437, 0.3339511752128601, 0.4302493631839752, 0.44165804982185364, 0.19940832257270813, 0.29347583651542664, 0.13689245283603668, 0.33998721837997437, 0.331850528717041, 0.258860319852829, 0.1024511530995369, 0.0463242344558239, 0.3564761281013489, 0.0020363135263323784, 0.3181179463863373, 0.23726680874824524, 0.3712978661060333, 0.06424477696418762, 0.1054452657699585, 0.24060940742492676, 0.14433854818344116], dtype='float32').reshape([24]),
            paddle.to_tensor([0.48519012331962585, 0.4975358247756958, 0.3693622946739197, 0.04142696410417557, 0.4873191714286804, 0.1974269598722458, 0.4720301926136017, 0.26236721873283386, 0.10473328083753586, 0.2444400042295456, 0.15042518079280853, 0.334561824798584, 0.0031463196501135826, 0.16767311096191406, 0.35956871509552, 0.3548411726951599, 0.326074481010437, 0.15529417991638184, 0.11866538971662521, 0.15334293246269226, 0.2492513507604599, 0.054638683795928955, 0.45392608642578125, 0.20398391783237457], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_53064740976ab337aaa59650e99accd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.026115378364920616, 0.3399978280067444, 0.18573975563049316, 0.1830003261566162, 0.12409113347530365, 0.44811078906059265, 0.3746044337749481, 0.05691293999552727, 0.46117648482322693, 0.21032361686229706, 0.34848594665527344, 0.42765700817108154, 0.48673659563064575, 0.33184465765953064, 0.2274269312620163, 0.07443918287754059, 0.19503997266292572, 0.11396331340074539, 0.19778768718242645, 0.35529500246047974, 0.20287920534610748, 0.47535407543182373, 0.4010811746120453, 0.4136248230934143], dtype='float32').reshape([24]),
            paddle.to_tensor([0.3828301727771759, 0.44412490725517273, 0.18499040603637695, 0.3435114026069641, 0.3052401542663574, 0.3681553304195404, 0.0995011106133461, 0.01962503232061863, 0.2231067717075348, 0.4545924961566925, 0.07231321930885315, 0.4412509500980377, 0.3204094469547272, 0.3284834921360016, 0.1406443864107132, 0.041892196983098984, 0.08240841329097748, 0.17574793100357056, 0.2553008198738098, 0.079107366502285, 0.3947557806968689, 0.4868345558643341, 0.4567123353481293, 0.4347275197505951], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9fe525d0a9c78c1abeb8872766d45743(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4758933484554291, 0.14255014061927795, 0.49187612533569336, 0.047913145273923874, 0.12605196237564087, 0.4230193793773651, 0.41245803236961365, 0.07384315133094788, 0.21088989078998566, 0.34581270813941956, 0.3833952844142914, 0.2822886109352112, 0.45838046073913574, 0.16176468133926392, 0.4614314138889313, 0.2802913784980774, 0.19649264216423035, 0.1444588452577591, 0.03105638176202774, 0.23455320298671722, 0.002982559846714139, 0.44490891695022583, 0.021438896656036377, 0.4839061498641968], dtype='float32').reshape([24]),
            paddle.to_tensor([0.051319241523742676, 0.4611882269382477, 0.22770775854587555, 0.11426018178462982, 0.3777332603931427, 0.31432104110717773, 0.46817252039909363, 0.2517809271812439, 0.11959251761436462, 0.1371832937002182, 0.015778006985783577, 0.13908152282238007, 0.005648371763527393, 0.015706151723861694, 0.4778655171394348, 0.11719158291816711, 0.22941698133945465, 0.04897738993167877, 0.10249515622854233, 0.40890949964523315, 0.4858309030532837, 0.10021840035915375, 0.4068313241004944, 0.25689175724983215], dtype='float32').reshape([24]),
        ]


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
class TestPrimitiveOp_97f9314656d27ce15624a4e6ea8d5d5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2873746156692505, 0.1815248727798462, 0.12061513215303421, 0.32241445779800415, 0.3212587237358093, 0.28378304839134216, 0.33587872982025146, 0.4938142001628876, 0.19452793896198273, 0.2163516730070114, 0.4053199887275696, 0.05240008607506752, 0.2817927598953247, 0.1612112820148468, 0.08422444760799408, 0.19342376291751862, 0.11218459159135818, 0.056300655007362366, 0.4635474681854248, 0.025622330605983734, 0.3040720820426941, 0.38445475697517395, 0.38712674379348755, 0.4090229868888855], dtype='float32').reshape([24]),
            paddle.to_tensor([0.18952342867851257, 0.2906690835952759, 0.04539253190159798, 0.15193432569503784, 0.24614083766937256, 0.15173020958900452, 0.4556560516357422, 0.4584605395793915, 0.2738528847694397, 0.28499531745910645, 0.2899218201637268, 0.10980573296546936, 0.3619408905506134, 0.30852559208869934, 0.4138665199279785, 0.11654220521450043, 0.35846373438835144, 0.2868020236492157, 0.1309565156698227, 0.22102117538452148, 0.001892277505248785, 0.25493401288986206, 0.21538372337818146, 0.41735348105430603], dtype='float32').reshape([24]),
        ]


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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_61d91043eb6cc3c42c786054a0cef1cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.009980613365769386, 0.2066096067428589, 0.052638985216617584, 0.08239495754241943, 0.008461661636829376, 0.37111377716064453, 0.11026058346033096, 0.09752855449914932, 0.2233685702085495, 0.15352359414100647, 0.23099084198474884, 0.005710940808057785, 0.4699607789516449, 0.22244541347026825, 0.08684197068214417, 0.44932010769844055, 0.10638318955898285, 0.1156611517071724, 0.25119829177856445, 0.05250447988510132, 0.12593379616737366, 0.0181891992688179, 0.05210256204009056, 0.4532413184642792], dtype='float32').reshape([24]),
            paddle.to_tensor([0.10958357900381088, 0.24095791578292847, 0.06223257631063461, 0.19261164963245392, 0.23442797362804413, 0.0922679752111435, 0.02529970370233059, 0.3026752173900604, 0.42767420411109924, 0.49469730257987976, 0.3415588438510895, 0.32167384028434753, 0.16551581025123596, 0.31206464767456055, 0.13808698952198029, 0.3272615671157837, 0.3406936526298523, 0.11879443377256393, 0.2191116213798523, 0.14108748733997345, 0.026989929378032684, 0.185555100440979, 0.4982188642024994, 0.12501585483551025], dtype='float32').reshape([24]),
        ]


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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_53702b2de8b6fdcfde17a25f022f7476(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3045271337032318, 0.36088189482688904, 0.4545067548751831, 0.422355055809021, 0.42839276790618896, 0.3346051275730133, 0.009054453112185001, 0.1608443260192871, 0.2292409986257553, 0.029927683994174004, 0.03858471289277077, 0.20469410717487335, 0.1728423535823822, 0.175882488489151, 0.39274173974990845, 0.26461121439933777, 0.3576749563217163, 0.10799141228199005, 0.19293850660324097, 0.05502169951796532, 0.05014834180474281, 0.2944068908691406, 0.3989642262458801, 0.3900720179080963], dtype='float32').reshape([24]),
            paddle.to_tensor([0.35645776987075806, 0.26274392008781433, 0.21716605126857758, 0.2551792562007904, 0.34680429100990295, 0.48434537649154663, 0.20857588946819305, 0.4435752034187317, 0.4403742849826813, 0.27582046389579773, 0.18716412782669067, 0.23656408488750458, 0.3588096499443054, 0.12757307291030884, 0.39947572350502014, 0.4014561176300049, 0.3294893503189087, 0.2672031819820404, 0.16525886952877045, 0.20853517949581146, 0.43141835927963257, 0.42793288826942444, 0.45923808217048645, 0.20088329911231995], dtype='float32').reshape([24]),
        ]


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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3752dc812af18fc11d492922d03382ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.125257208943367, 0.26630279421806335, 0.37516918778419495, 0.06796825677156448, 0.14348039031028748, 0.432963103055954, 0.23445843160152435, 0.255221426486969, 0.03421889990568161, 0.4957563579082489, 0.3195517957210541, 0.22233279049396515, 0.340664267539978, 0.24260041117668152, 0.10078512877225876, 0.3019268810749054, 0.4324286878108978, 0.3794633150100708, 0.2903705835342407, 0.2803431749343872, 0.058363426476716995, 0.19376982748508453, 0.46515753865242004, 0.21511000394821167], dtype='float32').reshape([24]),
            paddle.to_tensor([0.0890764519572258, 0.3386397957801819, 0.35162442922592163, 0.15555673837661743, 0.22990140318870544, 0.1320718675851822, 0.15941056609153748, 0.4153274595737457, 0.4542528986930847, 0.33819878101348877, 0.1382564902305603, 0.3610631823539734, 0.24435286223888397, 0.48980000615119934, 0.36535245180130005, 0.3917272984981537, 0.18642082810401917, 0.30590805411338806, 0.43089011311531067, 0.2654983103275299, 0.46121129393577576, 0.4739340543746948, 0.16264113783836365, 0.13679340481758118], dtype='float32').reshape([24]),
        ]


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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f90339ecdef9ec9ed0154002fc326dd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.22056183218955994, 0.3215359151363373, 0.46935907006263733, 0.24616269767284393, 0.022607149556279182, 0.17583952844142914, 0.45278456807136536, 0.33680304884910583, 0.28212082386016846, 0.10420001298189163, 0.16569019854068756, 0.4431293308734894, 0.38582777976989746, 0.04099467769265175, 0.24310868978500366, 0.39429986476898193, 0.02201562374830246, 0.09620098024606705, 0.29747137427330017, 0.11888834834098816, 0.4286400377750397, 0.33920061588287354, 0.1445033848285675, 0.13719695806503296], dtype='float32').reshape([24]),
            paddle.to_tensor([0.1128792092204094, 0.2592217028141022, 0.19194038212299347, 0.3717268109321594, 0.3226310908794403, 0.4229230582714081, 0.08634963631629944, 0.35607409477233887, 0.32209131121635437, 0.27355295419692993, 0.2836862802505493, 0.3129297196865082, 0.49770551919937134, 0.35394778847694397, 0.1329210102558136, 0.12742388248443604, 0.418045312166214, 0.2860706150531769, 0.29434463381767273, 0.01134207472205162, 0.4698030948638916, 0.2880990505218506, 0.15108497440814972, 0.029141156002879143], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0211085c603c8ee2c99b9ce330628ab1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.360409140586853, 0.08726335316896439, 0.12895233929157257, 0.35411882400512695, 0.3476443588733673, 0.4807546138763428, 0.46181726455688477, 0.43201878666877747, 0.002793261082842946, 0.1495034396648407, 0.4345085322856903, 0.2610032558441162, 0.21554253995418549, 0.059939850121736526, 0.4966655671596527, 0.45420312881469727, 0.43090003728866577, 0.0748104378581047, 0.08833171427249908, 0.3165273666381836, 0.06683310866355896, 0.34681040048599243, 0.33262521028518677, 0.16069728136062622], dtype='float32').reshape([24]),
            paddle.to_tensor([0.05348410829901695, 0.48318102955818176, 0.27209386229515076, 0.05398734658956528, 0.26978081464767456, 0.29747360944747925, 0.41010212898254395, 0.0545368492603302, 0.2075808197259903, 0.040098268538713455, 0.20714770257472992, 0.3729572296142578, 0.3550529479980469, 0.4786156117916107, 0.11409778892993927, 0.27507272362709045, 0.1889083832502365, 0.2074296623468399, 0.4609362781047821, 0.20381633937358856, 0.14024963974952698, 0.037127669900655746, 0.05402032285928726, 0.47654542326927185], dtype='float32').reshape([24]),
        ]


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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3a6e7cb2562fe2d4a188ad4fd723e79a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.42281147837638855, 0.08266928791999817, 0.3753553628921509, 0.35161104798316956, 0.10693258047103882, 0.29845544695854187, 0.1096993014216423, 0.001929763937368989, 0.1978520154953003, 0.39333727955818176, 0.3993414044380188, 0.06251654773950577, 0.002844803035259247, 0.3152347207069397, 0.34154224395751953, 0.27489545941352844, 0.4788269102573395, 0.3868279755115509, 0.029443953186273575, 0.1249895766377449, 0.3196967542171478, 0.400125116109848, 0.4868909418582916, 0.15367543697357178], dtype='float32').reshape([24]),
            paddle.to_tensor([0.15911105275154114, 0.3191523849964142, 0.30547034740448, 0.19786109030246735, 0.3050064444541931, 0.30814671516418457, 0.1367986649274826, 0.30087926983833313, 0.015460798516869545, 0.16673791408538818, 0.33956626057624817, 0.444155216217041, 0.24342040717601776, 0.4141845405101776, 0.3069738447666168, 0.4979444146156311, 0.014999326318502426, 0.28377583622932434, 0.40204620361328125, 0.44927504658699036, 0.1680687516927719, 0.4651588797569275, 0.02933969907462597, 0.2827579081058502], dtype='float32').reshape([24]),
        ]


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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d487722dce771ee7d284f53b86cf2644(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3612076938152313, 0.39912208914756775, 0.14872072637081146, 0.25858673453330994, 0.07486964017152786, 0.33277249336242676, 0.43904757499694824, 0.2137400209903717, 0.2100633680820465, 0.4653855562210083, 0.47798794507980347, 0.301695853471756, 0.3093136250972748, 0.2589452266693115, 0.31297364830970764, 0.1711631417274475, 0.16097794473171234, 0.3179294168949127, 0.009308655746281147, 0.30526605248451233, 0.4128403663635254, 0.3444308936595917, 0.40020883083343506, 0.37081053853034973], dtype='float32').reshape([24]),
            paddle.to_tensor([0.38822075724601746, 0.06748425215482712, 0.28732869029045105, 0.2620127201080322, 0.2130596786737442, 0.4589809775352478, 0.17724978923797607, 0.19458502531051636, 0.12818197906017303, 0.2951882481575012, 0.04652812331914902, 0.3143450617790222, 0.120363250374794, 0.45104241371154785, 0.19639159739017487, 0.06870115548372269, 0.34110501408576965, 0.008029647171497345, 0.08939842879772186, 0.08021437376737595, 0.34604668617248535, 0.37177780270576477, 0.26150017976760864, 0.43491631746292114], dtype='float32').reshape([24]),
        ]


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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cf0263796a597fa188e28cda0fd67319(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3243763744831085, 0.10922221839427948, 0.08571486920118332, 0.4509371221065521, 0.23479418456554413, 0.03712635114789009, 0.14491812884807587, 0.27291813492774963, 0.11994953453540802, 0.1481669545173645, 0.2928393483161926, 0.20127980411052704, 0.37972384691238403, 0.2028757780790329, 0.4817086458206177, 0.04140018671751022, 0.3429134786128998, 0.34433525800704956, 0.04517326503992081, 0.010226691141724586, 0.1278749257326126, 0.1376453936100006, 0.3240119218826294, 0.12344543635845184], dtype='float32').reshape([24]),
            paddle.to_tensor([0.4147098660469055, 0.15821808576583862, 0.145911306142807, 0.3174203634262085, 0.20298314094543457, 0.16132600605487823, 0.4538734257221222, 0.4587370455265045, 0.13601629436016083, 0.36636850237846375, 0.2857471704483032, 0.38322174549102783, 0.2860986888408661, 0.31123870611190796, 0.3196166157722473, 0.1634301245212555, 0.04482220858335495, 0.22947807610034943, 0.37294602394104004, 0.45078733563423157, 0.4548492729663849, 0.4942503869533539, 0.48787710070610046, 0.11250294744968414], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_64cc04383d27d7e4ebbd0743bc0375b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3483818769454956, 0.3278091847896576, 0.41863152384757996, 0.13953231275081635, 0.34160345792770386, 0.4341847002506256, 0.10421984642744064, 0.10705380141735077, 0.3194532096385956, 0.23210681974887848, 0.4578392803668976, 0.01700410805642605, 0.17198330163955688, 0.31888848543167114, 0.3827630579471588, 0.2457941174507141, 0.05894274264574051, 0.0029358668252825737, 0.13500957190990448, 0.4977151155471802, 0.28242582082748413, 0.10478909313678741, 0.13626962900161743, 0.18728817999362946], dtype='float32').reshape([24]),
            paddle.to_tensor([0.4803254306316376, 0.43768441677093506, 0.207551971077919, 0.2566898465156555, 0.4935014545917511, 0.45660173892974854, 0.0014975254889577627, 0.267597496509552, 0.13524766266345978, 0.08755500614643097, 0.24130550026893616, 0.009236403740942478, 0.13304661214351654, 0.11607170104980469, 0.24687431752681732, 0.49902647733688354, 0.3803535997867584, 0.005215034820139408, 0.45766791701316833, 0.043826859444379807, 0.3714032471179962, 0.0043741073459386826, 0.02027713693678379, 0.09170043468475342], dtype='float32').reshape([24]),
        ]


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
class TestPrimitiveOp_f74129ce47c2640da5552e0a2e4e1ee9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.40468478202819824, 0.20102308690547943, 0.21412764489650726, 0.34881195425987244, 0.15954531729221344, 0.3906097114086151, 0.41616368293762207, 0.2702464163303375, 0.2522966265678406, 0.19539105892181396, 0.34695497155189514, 0.49301624298095703, 0.4627770781517029, 0.037644173949956894, 0.4723157286643982, 0.36092641949653625, 0.2291240394115448, 0.21941101551055908, 0.3134632408618927, 0.39569878578186035, 0.04998055472970009, 0.3826083540916443, 0.31780657172203064, 0.08410269767045975], dtype='float32').reshape([24]),
            paddle.to_tensor([0.42166557908058167, 0.10020256787538528, 0.08045077323913574, 0.33110666275024414, 0.28873327374458313, 0.4704386591911316, 0.28217917680740356, 0.007594989612698555, 0.3561777174472809, 0.2565262019634247, 0.20871034264564514, 0.020144717767834663, 0.09739082306623459, 0.28244465589523315, 0.11376169323921204, 0.38699597120285034, 0.17603150010108948, 0.4838292598724365, 0.0001272702938877046, 0.1008094772696495, 0.1325010508298874, 0.3866124153137207, 0.13919435441493988, 0.395795077085495], dtype='float32').reshape([24]),
        ]


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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ac3ff9a2b5ca0ee94b422d5824f78279(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0210966058075428, 0.1405358612537384, 0.050669193267822266, 0.3962388336658478, 0.38744020462036133, 0.20083282887935638, 0.33778536319732666, 0.1588030755519867, 0.199144646525383, 0.33083635568618774, 0.18270771205425262, 0.46066367626190186, 0.17221054434776306, 0.12938076257705688, 0.32470399141311646, 0.2912845313549042, 0.3386044502258301, 0.24730175733566284, 0.08052926510572433, 0.2130158245563507, 0.1700984388589859, 0.26751574873924255, 0.37276071310043335, 0.44918128848075867], dtype='float32').reshape([24]),
            paddle.to_tensor([0.31895291805267334, 0.020774824544787407, 0.01710502803325653, 0.26318830251693726, 0.34087857604026794, 0.28989318013191223, 0.4662320911884308, 0.2854779064655304, 0.47583773732185364, 0.0061882007867097855, 0.30896711349487305, 0.007644944358617067, 0.018627822399139404, 0.45214036107063293, 0.05633212625980377, 0.0832122340798378, 0.3543674945831299, 0.29360178112983704, 0.3896822929382324, 0.018625479191541672, 0.4171549081802368, 0.46552833914756775, 0.08604802191257477, 0.21988141536712646], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_88f3fc49258c4485a00e2f3d211ab98c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.38297128677368164, 0.010262361727654934, 0.26849040389060974, 0.400585800409317, 0.023237742483615875, 0.33608707785606384, 0.49511146545410156, 0.42044952511787415, 0.14368371665477753, 0.2012222856283188, 0.4206011891365051, 0.02385561726987362, 0.0046110618859529495, 0.034425243735313416, 0.38097113370895386, 0.17182743549346924, 0.40061092376708984, 0.4605620801448822, 0.1295219212770462, 0.34910809993743896, 0.10261493176221848, 0.3077925145626068, 0.1498202383518219, 0.3907204568386078], dtype='float32').reshape([24]),
            paddle.to_tensor([0.10247986763715744, 0.4407726228237152, 0.37675759196281433, 0.11132825911045074, 0.009437788277864456, 0.011341910809278488, 0.3698680102825165, 0.016509748995304108, 0.17746783792972565, 0.1774677038192749, 0.01021084375679493, 0.3862353265285492, 0.0005669162492267787, 0.013267683796584606, 0.28642240166664124, 0.34295082092285156, 0.39364948868751526, 0.006558415479958057, 0.057813942432403564, 0.04779807850718498, 0.4497207999229431, 0.04904820770025253, 0.07234026491641998, 0.38010671734809875], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f5920f3963557ae0dd0ec9a121c4db73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2933265268802643, 0.16921186447143555, 0.01854425109922886, 0.487466424703598, 0.20632091164588928, 0.46411484479904175, 0.3132723569869995, 0.4928017258644104, 0.19609762728214264, 0.10990948975086212, 0.18288911879062653, 0.35555291175842285, 0.14033372700214386, 0.0720382034778595, 0.4805426299571991, 0.21052047610282898, 0.26630765199661255, 0.3649366796016693, 0.29626673460006714, 0.22779420018196106, 0.19369280338287354, 0.4335307478904724, 0.43144491314888, 0.49565356969833374], dtype='float32').reshape([24]),
            paddle.to_tensor([0.31373128294944763, 0.4824369549751282, 0.25017398595809937, 0.2534303069114685, 0.42491424083709717, 0.2757776379585266, 0.44201698899269104, 0.2650067210197449, 0.33668312430381775, 0.4967504143714905, 0.4213058650493622, 0.3587561547756195, 0.40281763672828674, 0.37119513750076294, 0.19323843717575073, 0.09903126955032349, 0.03534949570894241, 0.36447274684906006, 0.05788661167025566, 0.3030148148536682, 0.04634292051196098, 0.49821725487709045, 0.2200348973274231, 0.01853361912071705], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_54b7fef8cd622816bd0afba46c0cd0d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.015485916286706924, 0.3460428714752197, 0.41551101207733154, 0.06476900726556778, 0.17843209207057953, 0.4628934860229492, 0.1199483722448349, 0.26321372389793396, 0.2663916349411011, 0.11708617955446243, 0.047969717532396317, 0.05378718301653862, 0.27159857749938965, 0.47233375906944275, 0.4518211781978607, 0.37067002058029175, 0.4557918906211853, 0.045893386006355286, 0.4446979761123657, 0.40272200107574463, 0.25079548358917236, 0.28339874744415283, 0.1432686299085617, 0.44945579767227173], dtype='float32').reshape([24]),
            paddle.to_tensor([0.14335894584655762, 0.06016656011343002, 0.3126979470252991, 0.059239644557237625, 0.46179819107055664, 0.2539837956428528, 0.05213729664683342, 0.2744452953338623, 0.12494036555290222, 0.26780304312705994, 0.16777877509593964, 0.39798393845558167, 0.1627836525440216, 0.026539918035268784, 0.11397183686494827, 0.4219403862953186, 0.05174480751156807, 0.3475891649723053, 0.04770568758249283, 0.005230615381151438, 0.2112313061952591, 0.47769203782081604, 0.490946501493454, 0.11258085072040558], dtype='float32').reshape([24]),
        ]


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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_434980ff57b6007a98352aa539f2a11f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.09756629914045334, 0.09801867604255676, 0.30038389563560486, 0.45932257175445557, 0.07898318767547607, 0.007377479691058397, 0.20695184171199799, 0.33387717604637146, 0.032677263021469116, 0.18860885500907898, 0.021236848086118698, 0.03667052462697029, 0.41663211584091187, 0.3359481692314148, 0.25308653712272644, 0.32621538639068604, 0.13155707716941833, 0.4317314624786377, 0.22465412318706512, 0.35051223635673523, 0.3950556814670563, 0.2877831757068634, 0.04236484318971634, 0.34339913725852966], dtype='float32').reshape([24]),
            paddle.to_tensor([0.09727225452661514, 0.40842491388320923, 0.4322351813316345, 0.05528852716088295, 0.3019605875015259, 0.43542349338531494, 0.12367310374975204, 0.2120688408613205, 0.05228736996650696, 0.03723984584212303, 0.1830877959728241, 0.2512466311454773, 0.05638841539621353, 0.11866109073162079, 0.1470564901828766, 0.30375203490257263, 0.007476666010916233, 0.21952953934669495, 0.04817856848239899, 0.48219555616378784, 0.17738263309001923, 0.07973664999008179, 0.03273080289363861, 0.3500777781009674], dtype='float32').reshape([24]),
        ]


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
class TestPrimitiveOp_bc4742bfbf6ce09d98643fafdb820f5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.28283435106277466, 0.0676804631948471, 0.35826951265335083, 0.0053129130974411964, 0.23308177292346954, 0.05164271220564842, 0.39908045530319214, 0.43173283338546753, 0.45630308985710144, 0.40910929441452026, 0.4483509957790375, 0.14918763935565948, 0.2299223095178604, 0.40071001648902893, 0.25179004669189453, 0.49062275886535645, 0.3868352174758911, 0.1141994446516037, 0.2313208132982254, 0.45274618268013, 0.4197809398174286, 0.49996307492256165, 0.03991374745965004, 0.11095697432756424], dtype='float32').reshape([24]),
            paddle.to_tensor([0.15641798079013824, 0.2062762826681137, 0.3169999122619629, 0.11760962754487991, 0.4860123097896576, 0.41281992197036743, 0.07780268788337708, 0.3878086507320404, 0.09807993471622467, 0.19513367116451263, 0.3733014166355133, 0.34576311707496643, 0.4598023593425751, 0.08605511486530304, 0.3005611002445221, 0.3171677589416504, 0.11540449410676956, 0.453881174325943, 0.4378935992717743, 0.2092243731021881, 0.45806601643562317, 0.11833133548498154, 0.02251986414194107, 0.47287923097610474], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_370cd0a7b11543ffe1b62a9ad91cb690(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.22216951847076416, 0.4717188775539398, 0.4249148666858673, 0.27142882347106934, 0.1327742636203766, 0.28290072083473206, 0.4418167471885681, 0.11034416407346725, 0.4712463915348053, 0.49393007159233093, 0.3123990595340729, 0.1901472806930542, 0.08527329564094543, 0.3829624056816101, 0.29574015736579895, 0.011238908395171165, 0.284287691116333, 0.49747249484062195, 0.1427219808101654, 0.4186911880970001, 0.3912399709224701, 0.06630081683397293, 0.46670734882354736, 0.04763069748878479], dtype='float32').reshape([24]),
            paddle.to_tensor([0.0681866928935051, 0.16083848476409912, 0.22945405542850494, 0.35928285121917725, 0.12687727808952332, 0.4214862883090973, 0.42756494879722595, 0.27913698554039, 0.16280722618103027, 0.2662515342235565, 0.08638124912977219, 0.30440986156463623, 0.004144219681620598, 0.019412312656641006, 0.3725771903991699, 0.05502048879861832, 0.022154243662953377, 0.030937639996409416, 0.13815757632255554, 0.09306534379720688, 0.02362733706831932, 0.24921591579914093, 0.398305743932724, 0.20674848556518555], dtype='float32').reshape([24]),
        ]


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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5d9716eacf12f5841f27bd79b9195ae9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3460783362388611, 0.03443462774157524, 0.408433735370636, 0.48116007447242737, 0.16743560135364532, 0.085225909948349, 0.4584450125694275, 0.006579235196113586, 0.08808024972677231, 0.22657927870750427, 0.29153281450271606, 0.49339964985847473, 0.29302096366882324, 0.3056832253932953, 0.13836996257305145, 0.40149006247520447, 0.3809690773487091, 0.011306263506412506, 0.15014468133449554, 0.3011144995689392, 0.4817412197589874, 0.2781399190425873, 0.3353548049926758, 0.11786077916622162], dtype='float32').reshape([24]),
            paddle.to_tensor([0.15490134060382843, 0.3359456956386566, 0.1523849368095398, 0.45645374059677124, 0.41901591420173645, 0.23732852935791016, 0.12281060218811035, 0.38296806812286377, 0.06538064032793045, 0.30907142162323, 0.4211535155773163, 0.18149176239967346, 0.4729141592979431, 0.2542508840560913, 0.3233759105205536, 0.2626575827598572, 0.414181649684906, 0.22685237228870392, 0.017079800367355347, 0.2565867006778717, 0.18213969469070435, 0.15655970573425293, 0.4355728328227997, 0.16980616748332977], dtype='float32').reshape([24]),
        ]


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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2056867b932051e7198b29536c311b2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1823611706495285, 0.3962276875972748, 0.4280262887477875, 0.12127076089382172, 0.11854170262813568, 0.43120452761650085, 0.31985846161842346, 0.41223856806755066, 0.3951638340950012, 0.4401490390300751, 0.4098871350288391, 0.12045428901910782, 0.36951667070388794, 0.3026745915412903, 0.18505096435546875, 0.007786941714584827, 0.24096012115478516, 0.12980540096759796, 0.49633386731147766, 0.3096555769443512, 0.2751445472240448, 0.18698763847351074, 0.4963749349117279, 0.4187670052051544], dtype='float32').reshape([24]),
            paddle.to_tensor([0.2842351198196411, 0.029225170612335205, 0.03564634174108505, 0.26090165972709656, 0.22109831869602203, 0.4564916789531708, 0.053904809057712555, 0.1938057243824005, 0.29531407356262207, 0.3922020494937897, 0.24451473355293274, 0.39571478962898254, 0.28309890627861023, 0.28385430574417114, 0.4339904189109802, 0.34302595257759094, 0.009946681559085846, 0.268003910779953, 0.0916522666811943, 0.3937080204486847, 0.4599798023700714, 0.009313804097473621, 0.4707798659801483, 0.45024019479751587], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0fa34a9c352b0f0017eb38b39ecbd7cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3191579580307007, 0.2731638550758362, 0.3224371075630188, 0.031068051233887672, 0.1186930239200592, 0.3118444085121155, 0.40523266792297363, 0.40708720684051514, 0.10220556706190109, 0.41578182578086853, 0.47054848074913025, 0.3298671543598175, 0.23831669986248016, 0.18965032696723938, 0.4976508319377899, 0.23149944841861725, 0.2474496215581894, 0.2674003839492798, 0.17953433096408844, 0.4432717263698578, 0.35693398118019104, 0.4385075867176056, 0.32127657532691956, 0.460958868265152], dtype='float32').reshape([24]),
            paddle.to_tensor([0.007759390398859978, 0.39703166484832764, 0.14795221388339996, 0.20501507818698883, 0.07570552080869675, 0.33046460151672363, 0.053221795707941055, 0.1448243260383606, 0.39755159616470337, 0.1425670087337494, 0.020017744973301888, 0.2278515249490738, 0.03960034251213074, 0.21489393711090088, 0.1418943703174591, 0.4585473835468292, 0.40421321988105774, 0.12636959552764893, 0.315336674451828, 0.25199800729751587, 0.4459870457649231, 0.1506226360797882, 0.12882934510707855, 0.25234848260879517], dtype='float32').reshape([24]),
        ]


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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_57185e340d1c77b73556f4956e9600ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.27327877283096313, 0.3870028853416443, 0.06256396323442459, 0.02868662402033806, 0.3612062633037567, 0.12573139369487762, 0.011210931465029716, 0.36385631561279297, 0.008875143714249134, 0.16479448974132538, 0.4837847054004669, 0.20957621932029724, 0.1223117932677269, 0.09353309124708176, 0.2946908175945282, 0.40447285771369934, 0.25413790345191956, 0.07305101305246353, 0.4619729220867157, 0.09364607185125351, 0.08762020617723465, 0.2246808409690857, 0.4245934784412384, 0.018741071224212646], dtype='float32').reshape([24]),
            paddle.to_tensor([0.16428643465042114, 0.008446759544312954, 0.14582963287830353, 0.13918288052082062, 0.26555201411247253, 0.2110971063375473, 0.4846075773239136, 0.050534699112176895, 0.1081136092543602, 0.2533041834831238, 0.16036318242549896, 0.2517251968383789, 0.4891287088394165, 0.10519136488437653, 0.491793692111969, 0.3665371239185333, 0.321613073348999, 0.3481799066066742, 0.3071436882019043, 0.4702399969100952, 0.0004998861695639789, 0.044105228036642075, 0.440019428730011, 0.4786377251148224], dtype='float32').reshape([24]),
        ]


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
class TestPrimitiveOp_49dd439c1212eb15f1da96a1157a2de5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.35693100094795227, 0.49161261320114136, 0.3728858232498169, 0.06533476710319519, 0.31946367025375366, 0.22106540203094482, 0.04963909462094307, 0.46916940808296204, 0.2829035222530365, 0.06328136473894119, 0.13949783146381378, 0.060443803668022156, 0.23511122167110443, 0.4260987341403961, 0.47255101799964905, 0.12235407531261444, 0.4374164342880249, 0.24069102108478546, 0.3153023421764374, 0.1330728679895401, 0.0725119411945343, 0.21814101934432983, 0.14752711355686188, 0.12983427941799164], dtype='float32').reshape([24]),
            paddle.to_tensor([0.022101860493421555, 0.3643698990345001, 0.14877784252166748, 0.008698607794940472, 0.20557324588298798, 0.27467259764671326, 0.04583905637264252, 0.08336306363344193, 0.40658047795295715, 0.2563936710357666, 0.4374331831932068, 0.1845434606075287, 0.3933418095111847, 0.16926386952400208, 0.4192522168159485, 0.4040462374687195, 0.049375273287296295, 0.04602387920022011, 0.3717819154262543, 0.02422940731048584, 0.49567240476608276, 0.33613118529319763, 0.365788072347641, 0.136431023478508], dtype='float32').reshape([24]),
        ]


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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ee20a11c262818f45fede28598772e28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.06304154545068741, 0.29864418506622314, 0.1388540416955948, 0.39037221670150757, 0.30445072054862976, 0.13776838779449463, 0.43298131227493286, 0.4721490144729614, 0.17988599836826324, 0.46066993474960327, 0.24454152584075928, 0.3099138140678406, 0.018890922889113426, 0.16745896637439728, 0.023814519867300987, 0.29015523195266724, 0.4948238134384155, 0.442834734916687, 0.43423059582710266, 0.22957034409046173, 0.4154045283794403, 0.01994956284761429, 0.4893243610858917, 0.21568860113620758], dtype='float32').reshape([24]),
            paddle.to_tensor([0.06159462779760361, 0.17987963557243347, 0.2939353585243225, 0.40622374415397644, 0.016140853986144066, 0.010005838237702847, 0.3262770473957062, 0.04786226153373718, 0.006287831813097, 0.1692119538784027, 0.439468115568161, 0.35572150349617004, 0.45436128973960876, 0.3452468812465668, 0.4358096420764923, 0.358969122171402, 0.30024629831314087, 0.08428844809532166, 0.4258826673030853, 0.24552342295646667, 0.21239078044891357, 0.4502423405647278, 0.4897916316986084, 0.24511180818080902], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ce6223697b9603d06af0393ba16b5cc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2686774730682373, 0.015429876744747162, 0.008320430293679237, 0.2537198066711426, 0.37676122784614563, 0.3731080889701843, 0.05977616831660271, 0.0696033164858818, 0.4978320598602295, 0.3556686043739319, 0.46474823355674744, 0.18952178955078125, 0.3007684648036957, 0.1707219034433365, 0.4185183048248291, 0.15997563302516937, 0.1978771984577179, 0.09467022120952606, 0.35740023851394653, 0.15861745178699493, 0.39399126172065735, 0.44170236587524414, 0.1843089908361435, 0.1804286390542984], dtype='float32').reshape([24]),
            paddle.to_tensor([0.4174814820289612, 0.35215041041374207, 0.34631291031837463, 0.02836156263947487, 0.4176118075847626, 0.4666920006275177, 0.31074315309524536, 0.45024096965789795, 0.30359527468681335, 0.07473526895046234, 0.01515351515263319, 0.46556949615478516, 0.15265941619873047, 0.22525517642498016, 0.008867619559168816, 0.11759386956691742, 0.34550464153289795, 0.21290560066699982, 0.4914778470993042, 0.2836286425590515, 0.057972781360149384, 0.27428141236305237, 0.26461225748062134, 0.05772741138935089], dtype='float32').reshape([24]),
        ]


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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_01af856556a4d5960fe4f496216b583c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4881134033203125, 0.11468533426523209, 0.2612071931362152, 0.38030245900154114, 0.021396568045020103, 0.45309773087501526, 0.16356751322746277, 0.11987242102622986, 0.11192021518945694, 0.027293849736452103, 0.47223973274230957, 0.47053709626197815, 0.0325610525906086, 0.06356102228164673, 0.3113722801208496, 0.0007756504346616566, 0.08912140876054764, 0.4233382046222687, 0.03237156569957733, 0.36110302805900574, 0.022093860432505608, 0.1379624754190445, 0.27979791164398193, 0.4354380667209625], dtype='float32').reshape([24]),
            paddle.to_tensor([0.1689424365758896, 0.30405688285827637, 0.38950446248054504, 0.16508503258228302, 0.3236425518989563, 0.2509315311908722, 0.15311509370803833, 0.40804144740104675, 0.42480573058128357, 0.024796435609459877, 0.14060787856578827, 0.002965668449178338, 0.42817965149879456, 0.4381483793258667, 0.2640034258365631, 0.23062065243721008, 0.03668816760182381, 0.3544042110443115, 0.30383992195129395, 0.16835841536521912, 0.2993701696395874, 0.004579231142997742, 0.40677696466445923, 0.3733751177787781], dtype='float32').reshape([24]),
        ]


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
class TestPrimitiveOp_8af5ef62517ed847b00492d27371c2f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.12513338029384613, 0.18403656780719757, 0.27308550477027893, 0.42414334416389465, 0.17093493044376373, 0.14989915490150452, 0.37266695499420166, 0.28934377431869507, 0.00896511971950531, 0.1469772905111313, 0.3187383711338043, 0.05427708849310875, 0.294699490070343, 0.20837841928005219, 0.49950119853019714, 0.060025960206985474, 0.376433789730072, 0.0905025452375412, 0.1500069797039032, 0.12956218421459198, 0.3482898771762848, 0.34389713406562805, 0.46167463064193726, 0.42420631647109985], dtype='float32').reshape([24]),
            paddle.to_tensor([0.39863619208335876, 0.37510836124420166, 0.01863253489136696, 0.40454208850860596, 0.3168427646160126, 0.33005544543266296, 0.22674895823001862, 0.33450809121131897, 0.4460521638393402, 0.3933388292789459, 0.4529915153980255, 0.23362481594085693, 0.07341182231903076, 0.38512852787971497, 0.38534075021743774, 0.35783013701438904, 0.4853615164756775, 0.4360749125480652, 0.16441334784030914, 0.10529271513223648, 0.31381669640541077, 0.40715497732162476, 0.3519487679004669, 0.08888618648052216], dtype='float32').reshape([24]),
        ]


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
class TestPrimitiveOp_4ad2ada20b6ffc890c3808310c17a2e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.07127968221902847, 0.09125033766031265, 0.023202193900942802, 0.3497844934463501, 0.3141830563545227, 0.2902604937553406, 0.15621709823608398, 0.1674126237630844, 0.16408847272396088, 0.24759742617607117, 0.042430564761161804, 0.10406863689422607, 0.4620820879936218, 0.020261798053979874, 0.1034533753991127, 0.3168523907661438, 0.0856364369392395, 0.38661009073257446, 0.30293703079223633, 0.4134373366832733, 0.005359933711588383, 0.21869277954101562, 0.3607705533504486, 0.34248507022857666], dtype='float32').reshape([24]),
            paddle.to_tensor([0.09426507353782654, 0.3384673297405243, 0.2706443667411804, 0.3504253923892975, 0.1125011295080185, 0.1436195820569992, 0.05829785764217377, 0.0059114666655659676, 0.1310701221227646, 0.21837809681892395, 0.19304171204566956, 0.14124375581741333, 0.021088536828756332, 0.256740003824234, 0.01590237393975258, 0.3753992021083832, 0.3164536952972412, 0.05302175134420395, 0.09527934342622757, 0.406658411026001, 0.3974762260913849, 0.2860569357872009, 0.491249680519104, 0.11801835894584656], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e99b0c7132891699d9f9661ca322bb96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.40648627281188965, 0.24538792669773102, 0.48398125171661377, 0.42668208479881287, 0.22706156969070435, 0.21326547861099243, 0.04420938715338707, 0.4704393744468689, 0.23162122070789337, 0.2600278854370117, 0.23639260232448578, 0.09334257245063782, 0.43995553255081177, 0.34260687232017517, 0.04798019304871559, 0.3543713688850403, 0.15904739499092102, 0.13867199420928955, 0.20997390151023865, 0.15397725999355316, 0.1087106391787529, 0.2716803550720215, 0.1969616860151291, 0.2087760865688324], dtype='float32').reshape([24]),
            paddle.to_tensor([0.30893054604530334, 0.20366156101226807, 0.42002734541893005, 0.48796215653419495, 0.029658503830432892, 0.0772138461470604, 0.0037884870544075966, 0.24181978404521942, 0.28350767493247986, 0.28913992643356323, 0.4851824641227722, 0.18901866674423218, 0.0048148250207304955, 0.4520302414894104, 0.1805366426706314, 0.3485776484012604, 0.013709526509046555, 0.36075249314308167, 0.057362738996744156, 0.03830808401107788, 0.1693912297487259, 0.27795732021331787, 0.04746037721633911, 0.28284844756126404], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ae98ef3778406591e89e7cc23e9059e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4089096188545227, 0.1885455697774887, 0.06229705363512039, 0.025573620572686195, 0.40406835079193115, 0.3970227539539337, 0.31646642088890076, 0.4974219799041748, 0.48800355195999146, 0.19977805018424988, 0.4191742539405823, 0.4992269277572632, 0.1220344752073288, 0.3901118338108063, 0.47217321395874023, 0.3088521957397461, 0.04744509980082512, 0.19158270955085754, 0.4859465956687927, 0.00586653221398592, 0.011697019450366497, 0.03876720741391182, 0.314231812953949, 0.49260035157203674], dtype='float32').reshape([24]),
            paddle.to_tensor([0.29054465889930725, 0.20007146894931793, 0.0699198991060257, 0.08059374988079071, 0.46132805943489075, 0.3376687467098236, 0.4251303970813751, 0.4897180199623108, 0.22213463485240936, 0.0775279775261879, 0.4908750653266907, 0.3188328742980957, 0.3673385679721832, 0.29833224415779114, 0.4389877915382385, 0.2682447135448456, 0.396658718585968, 0.15508116781711578, 0.0841103047132492, 0.10179045051336288, 0.10187768191099167, 0.48471173644065857, 0.21435533463954926, 0.06461380422115326], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_083d37d0bffd6da953be64896f8d929f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4217832386493683, 0.26176580786705017, 0.42301103472709656, 0.06899531185626984, 0.49602118134498596, 0.23967638611793518, 0.17430457472801208, 0.4637378752231598, 0.13974443078041077, 0.07450118660926819, 0.011908818036317825, 0.49217328429222107, 0.30321693420410156, 0.3209482729434967, 0.2737846374511719, 0.4105992317199707, 0.4389213025569916, 0.3283942937850952, 0.02113870345056057, 0.05370422452688217, 0.04829084128141403, 0.494429349899292, 0.3366137742996216, 0.07904564589262009], dtype='float32').reshape([24]),
            paddle.to_tensor([0.10020922124385834, 0.36040979623794556, 0.3451809585094452, 0.08937521278858185, 0.04954676330089569, 0.2155519723892212, 0.06478944420814514, 0.019300175830721855, 0.2674526870250702, 0.4500877261161804, 0.2251822054386139, 0.2585299015045166, 0.10121611505746841, 0.40257906913757324, 0.30198734998703003, 0.30425578355789185, 0.37847697734832764, 0.3676827847957611, 0.09500036388635635, 0.2978985011577606, 0.3220682144165039, 0.31413328647613525, 0.20751874148845673, 0.12527833878993988], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_61862123d57a3f54f7ef877ff37ffb5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.41577062010765076, 0.4890482425689697, 0.03013896569609642, 0.3531072437763214, 0.3160388171672821, 0.24565263092517853, 0.07122120261192322, 0.3650830388069153, 0.2216411679983139, 0.23102734982967377, 0.09765178710222244, 0.4567965567111969, 0.1789235770702362, 0.3257059156894684, 0.0029165558516979218, 0.3824552595615387, 0.11337274312973022, 0.4016767740249634, 0.4744713604450226, 0.4593200087547302, 0.16416361927986145, 0.13903988897800446, 0.47507229447364807, 0.37058204412460327], dtype='float32').reshape([24]),
            paddle.to_tensor([0.03755776584148407, 0.21775440871715546, 0.23867663741111755, 0.3571220338344574, 0.0036783162504434586, 0.3568361699581146, 0.3765510320663452, 0.39102163910865784, 0.44721299409866333, 0.012020466849207878, 0.3160547912120819, 0.029246484860777855, 0.27084386348724365, 0.11951540410518646, 0.041485194116830826, 0.4464377164840698, 0.33930811285972595, 0.4862894117832184, 0.24695613980293274, 0.27994653582572937, 0.03206433728337288, 0.3237987160682678, 0.2470344603061676, 0.34010496735572815], dtype='float32').reshape([24]),
        ]


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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d8c70c93550090999f00d8ddb5f9b4cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.24747993052005768, 0.08444679528474808, 0.4609930217266083, 0.3207951486110687, 0.4351491332054138, 0.3688061535358429, 0.2839893102645874, 0.3300152122974396, 0.41294193267822266, 0.3649066090583801, 0.3473729193210602, 0.4198988378047943, 0.1625453680753708, 0.08249597996473312, 0.38573628664016724, 0.294180303812027, 0.19753725826740265, 0.08508450537919998, 0.4258096218109131, 0.2672591209411621, 0.39229899644851685, 0.4606698751449585, 0.2101658284664154, 0.4031069576740265], dtype='float32').reshape([24]),
            paddle.to_tensor([0.0938786193728447, 0.30581432580947876, 0.01788550801575184, 0.05210574343800545, 0.31671133637428284, 0.10273763537406921, 0.2759387195110321, 0.4677678346633911, 0.2512931525707245, 0.18504437804222107, 0.06472693383693695, 0.43672898411750793, 0.38604217767715454, 0.07011484354734421, 0.4461390972137451, 0.49022847414016724, 0.06516890972852707, 0.4210980236530304, 0.428902268409729, 0.08478165417909622, 0.22496168315410614, 0.34644636511802673, 0.078738734126091, 0.30314740538597107], dtype='float32').reshape([24]),
        ]


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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bc4a5a121648b39e041a7c9ea0480cc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4347810447216034, 0.15940269827842712, 0.17664343118667603, 0.3824494481086731, 0.24352580308914185, 0.33991098403930664, 0.42089536786079407, 0.04446713998913765, 0.3208412230014801, 0.33512669801712036, 0.21370652318000793, 0.04180944338440895, 0.03875293955206871, 0.17902922630310059, 0.09477446973323822, 0.20786434412002563, 0.22030729055404663, 0.462057888507843, 0.288800984621048, 0.21179451048374176, 0.030562235042452812, 0.4673629701137543, 0.3987131416797638, 0.151397705078125], dtype='float32').reshape([24]),
            paddle.to_tensor([0.12115351110696793, 0.08800000697374344, 0.20509029924869537, 0.4517628848552704, 0.41291823983192444, 0.14662465453147888, 0.3803623616695404, 0.10470513254404068, 0.4940492808818817, 0.20152774453163147, 0.4801822304725647, 0.4221990704536438, 0.2589115798473358, 0.14514170587062836, 0.3002837896347046, 0.03473708778619766, 0.10798722505569458, 0.4809415340423584, 0.22394360601902008, 0.37412530183792114, 0.04837535694241524, 0.493552565574646, 0.35049429535865784, 0.0923210084438324], dtype='float32').reshape([24]),
        ]


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