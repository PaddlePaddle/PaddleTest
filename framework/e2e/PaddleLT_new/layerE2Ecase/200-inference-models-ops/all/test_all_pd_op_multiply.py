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
class PrimitiveOp_98b6577781eac665e6737cbfd8d3b72c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_26955f968f9e688fae0d5ceda2e9787a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98b6577781eac665e6737cbfd8d3b72c
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
            paddle.to_tensor([160], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_90135cce2357169f6db23af094f48dab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_398c40626f96ab956d67eb5aee694815(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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
class TestPrimitiveOp_95cf6ac8f839963bb4a966e8f587d589(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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

class PrimitiveOp_084a8fc7f78f34762dbe8c9b3332dfe5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cdf2454f20ed5b7a9eb0164e100d3427(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_084a8fc7f78f34762dbe8c9b3332dfe5
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

class PrimitiveOp_02b9d20c36611ef2fad29beb0243d068(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_20e3fadf5a909794ffa38315a60f08bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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
class TestPrimitiveOp_176762d711e7408e7d9f003d9e028918(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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
class TestPrimitiveOp_78db386763ecb3c5964c0b49ed28e63e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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
class TestPrimitiveOp_fdae117ef2cb1e466c007a0730fabf8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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
class TestPrimitiveOp_27f336f91ef407c428bf3f8ebc7164c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98b6577781eac665e6737cbfd8d3b72c
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
            paddle.to_tensor([240], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_21e3d3ba99c91a8cdffbf537f64ee68f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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
class TestPrimitiveOp_c45b10bdd9cd8450fc1715d1e3508fac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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
class TestPrimitiveOp_5fb4f2363c97a9224d9f6cd8d4d17914(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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
class TestPrimitiveOp_0ea79712ff74c6b75592baf0eefe30d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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
class TestPrimitiveOp_192c243a4d6f2d07667ad63bc8b67156(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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
class TestPrimitiveOp_98af5f852598b6eeb8e3711fc32b0961(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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
class TestPrimitiveOp_862709e4458437e3e57e2ed5b0299ed2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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
class TestPrimitiveOp_750700295b5736232356262b5a4c716a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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
class TestPrimitiveOp_a8ea5cb78cfa4778fddf3a439a5871a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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
class TestPrimitiveOp_bf2ec4d92d4a9bd9e45f75d57b00e360(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 800, 1344], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([[[[0.2685546875]], [[0.08740234375]], [[0.287353515625]]]], dtype='float16').reshape([1, 3, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ab7aa7e1c6670072178d2873de46a805(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9f0e84f506258b9aec557e520df30a66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab7aa7e1c6670072178d2873de46a805
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 100, 168], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.311279296875], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0d186af54a9443390c6636c47f525980(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab7aa7e1c6670072178d2873de46a805
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 50, 84], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.482177734375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_187597dcbd8969fac6020d1cecab7ee9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab7aa7e1c6670072178d2873de46a805
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 25, 42], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.303955078125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d3e4742781db4bff64f9c12df31e9052(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab7aa7e1c6670072178d2873de46a805
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 13, 21], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.14013671875], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d64075d8416ede6b267754f316b66776(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab7aa7e1c6670072178d2873de46a805
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 7, 11], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.03875732421875], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_968c2c05f1d268aff48736b85ec4c230(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_084a8fc7f78f34762dbe8c9b3332dfe5
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
class TestPrimitiveOp_43329248a316924fbb5c51b064c39133(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_084a8fc7f78f34762dbe8c9b3332dfe5
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
class TestPrimitiveOp_ae34696307045a4bf0905108e319ee70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_084a8fc7f78f34762dbe8c9b3332dfe5
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
class TestPrimitiveOp_8d02acb150ea396be391b78eaa93d08b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_084a8fc7f78f34762dbe8c9b3332dfe5
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
class TestPrimitiveOp_6eec3c3b1cbd98e99fc8feb8326f416c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_084a8fc7f78f34762dbe8c9b3332dfe5
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
class TestPrimitiveOp_00bbdf8d91eb936ec38c39c649210d38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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
class TestPrimitiveOp_eeba78c9886c715836df5556bb7ac750(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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
class TestPrimitiveOp_42f80d2eae317c470c3da95dbe22c627(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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
class TestPrimitiveOp_1415daf276a1170967d2661050359dab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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
class TestPrimitiveOp_81c951b6b2d3c75cc637d289a7989d6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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
class TestPrimitiveOp_ecef7446d7b56b128cdf16284a29e06c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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

class PrimitiveOp_8b3c803c1e709e58a0c64ac9b43e2021(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5aeee863f5801ff906d04868cfbbccbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b3c803c1e709e58a0c64ac9b43e2021
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
class TestPrimitiveOp_105a99e08adfd41b733343020722ee3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b3c803c1e709e58a0c64ac9b43e2021
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
class TestPrimitiveOp_8a074b97cede36851686069eee3b8478(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b3c803c1e709e58a0c64ac9b43e2021
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
class TestPrimitiveOp_8cee368fa7bd8b9b446b4b5dd0f4cc45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b3c803c1e709e58a0c64ac9b43e2021
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
class TestPrimitiveOp_257275e7ac87cdbc2388cea07edda166(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b3c803c1e709e58a0c64ac9b43e2021
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

class PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2868c645dd251495140b6c840e8255ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.034269895404577255], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_f456df1e4d858391e0e0fdee55703f5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3553077280521393], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_f43fd1378d0b322bacce228f61db5e70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.45771870017051697], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_8b73bac492f6bb49251988093ad4b3af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.025557570159435272], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_128b948602bdfa20177a5f73e38de18d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4053143262863159], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_233d568021ccb7a0b70c61714232f778(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.13564513623714447], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_1361a5b572d342fd4d70d044cec6f0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.01814817450940609], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_ca71788fffa3736d8c2fea8581e3080d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.21919342875480652], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_d1e37bd3fa7c0f945fe49e43910b0263(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2764962315559387], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_29b4fa2f6cdfaefdb40cf1f633a02533(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2401210069656372], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_d02ebc54dbcc3db22be740c2cdf8b449(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3128359317779541], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_a6911c3423dd03547dbeb4063f76388d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0034283464774489403], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_fc1cb87095f34cacc8e5098c6f095d2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.11662230640649796], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_17e81cfdfc8dcf145788aee160202502(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.47155091166496277], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_aa91fd746a0c13913e953a7fcadf2c6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2890031635761261], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_ef274fd541a81d5a36c51db5c885763e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3942105770111084], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_020a5800fd3ace24bbf00558ad8e0e06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.25721466541290283], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_d5966730a38d43927a84bc35a02376eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2686125636100769], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_1b863562f48116865157a97357fbec70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.23588304221630096], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_eabfda7acfd016d978a275b731b471d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.07459518313407898], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_3c9a0f41f0e77f1ccc74532a511e8990(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.07328919321298599], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_c65946df9da32c4f13902737679cfc71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1305624097585678], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_3ea6572231ea06e44c0ebb9ff617911c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.12345968931913376], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_3fea4cdd638d63fb8cf1d2406451f0ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.32259705662727356], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_0263e0701f6ef43c2fb0157260959ae3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2142874300479889], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_5d7a50b76a1664db8899f907df2b4e4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.033593520522117615], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_c6b53ed9121ae4ad202b4274b8ffd85a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.33284541964530945], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_d39e3861de7db5748a1fe062ab161caf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.09706348925828934], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_f08b4f532251ef8a5b61f1045e70e522(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.16094255447387695], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_fdffcc7ec66463791831676adcc6f2f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4290601909160614], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_c8c39097ebaa5818b66aae6428b73bfe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.055179841816425323], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_d883cdae4135ce8f9e3c9273b11692cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.042275089770555496], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_ae417343875fa00beab823f58143edee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.061161477118730545], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_67d12f0fa8fcba3ea49449de4e64d184(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3398377597332001], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_d55134feeb49117d9850a6b0f5c8f24d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3320675194263458], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_f231e5497ec920c2358759f3c8590965(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.44745224714279175], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_b9dfa84bf07f34a317d7ff2398880823(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3121994137763977], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_66db21a6539377cc6e9c195fc75dc8d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10801368206739426], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_d55589cfb8ed3e32a4ed3d5b95e230b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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
class TestPrimitiveOp_9a1bcb078d8c6a14bb6db412c20aabb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3038066625595093], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_03d6f660d6f55aa9600a21cfe8df8d9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.23026257753372192], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_dce15b534d4abddec8a518299217c851(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.09069259464740753], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_8e7bab5fa3034a15220217f8b5415395(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3072358965873718], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_a06adff2d93c90fb7d5c932b33613877(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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
class TestPrimitiveOp_e697bade39293a4aa1bcd10cb9116a1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.17319104075431824], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_e9cb403d71d97ef54a5d962ab283827f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.014830023981630802], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_eb67a197ea6c91c3c39eac55f8b06f1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.24390298128128052], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_b4fcd3bc283705270b3987a17c661a0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.25205427408218384], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_2573dda89795426330793c13b4c5c5bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.19475553929805756], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_5779d6aa31a5c441ac39d69d39545fe8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.21722935140132904], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_b0a492fc2e86189f31b535129f9a1711(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3297799527645111], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_95d943f17f2f5836faba819e723c96ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4012543857097626], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_f5be1f18b6498213c226bbbc637354be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2193007469177246], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_c1addfdb8d3be0bbba4f9f211e6edb7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3298938572406769], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_84b7de57cb74c6da8c94b7d3e7f201d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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
class TestPrimitiveOp_443c5803ec2e5a48ba5357d29d1c6277(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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
class TestPrimitiveOp_c3bbeab6244727432b7945e88e834856(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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
class TestPrimitiveOp_692f4ce3b2132db2dace18093f34f3a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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
class TestPrimitiveOp_207aeef3780d0c58e0d46bb7b187e9e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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
class TestPrimitiveOp_1475f71f35eeae73665ab8482c79b3c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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
class TestPrimitiveOp_708f0224738c442be3e3a7b2c79a6059(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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
class TestPrimitiveOp_dca7a5ceb0779c7182e4e4541c9ce003(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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

class PrimitiveOp_8ba92b1846673267f0288e09a29d9ccb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1d1191e8ff4abafbf52bd6e7bc9cff62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ba92b1846673267f0288e09a29d9ccb
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
class TestPrimitiveOp_bccac161a89381daa602207ef677a5a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ba92b1846673267f0288e09a29d9ccb
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
class TestPrimitiveOp_6bb2a548e11e7e26cc132415f6a3c5ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ba92b1846673267f0288e09a29d9ccb
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
class TestPrimitiveOp_b012c4607a19cf29042bbc81727b7c3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ba92b1846673267f0288e09a29d9ccb
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
class TestPrimitiveOp_06b756283b8631faee6459d699744d05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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
class TestPrimitiveOp_1a40961227b879752d5a1e1fb9a76ade(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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
class TestPrimitiveOp_ceb2c1111253e8ebe206f22a1cbea511(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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
class TestPrimitiveOp_2880f4eb97173e54c1b143d4a62f17de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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
class TestPrimitiveOp_2650bfbdc2c2707970c1686cd484f89f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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
class TestPrimitiveOp_679997ce33ef6addaa98f8e69909fe77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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

class PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_19d8b7548fbc4d6302412fc2c1d6ada0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.156005859375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_dd35fc29662e0d63cfadd89552dc21d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.423583984375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_022f4226b8ef0cdd16a2170f0d323c73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1646728515625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_1c1337cebe406823a9be5710fc8ad702(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.386962890625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_99e68f41b3e63211e8b57f8616fa0f21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.33349609375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_9c8ad72a8b7af8c65ecfd019004fdf7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2442626953125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_77b87fe34dfbbfbf885be547b9984b5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.26318359375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_d814ce8bc05525f8758272844169ada9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.04327392578125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_f35d618a1119f9c4a5ca8adb06bf48c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.12042236328125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_3a3707f3d407563a5473ac901fa667d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.202392578125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_5680d814ba4f11ed0ddb771805677a6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1036376953125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_f7373c5bce24cf0fe9b09b01add64a8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.134521484375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_42160198d35e3500c007aab949028cf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.37158203125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_1bf8c50d5b1d0065097f734b73c4002f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0194549560546875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_376826526d01f66ca713169751b7a9b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.12030029296875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_a97d22089b7ac76acc49e46d28e16fe0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3505859375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_e25ca3a2d8182f184869cee1dbc3a7f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1939697265625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_b78e869834a965f9c7346b38ee103d72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.312255859375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_8b70e4fd9e5187825bca6765fab181de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.020751953125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_26c45d72b1115b4e4780b5089125f80e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.41552734375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_19e3979c9430d770217e539e5d09d1d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2044677734375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_e5b5daedb2d0139ae84bc721f27738d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.47998046875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_39310af8c807833219c769ac4699d499(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2447509765625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_7345c0fe74daca3cce68934666aee6a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.23974609375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_ff7f193f2b9277096a56087d66184bea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.21630859375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_b079aa9029a87733208d24e5f090c4b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.213623046875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_d717bb9c162ee2c5d78889d1e1c2a21d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.36865234375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_af5163627ca0c2b1b0f9fda59d8dbd20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.39111328125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_8b6332a48fd992d7ab4fc0334b2a21bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2152099609375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_a7e0e465c10bff0ddbe1554ae50bcc1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.25], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_e5240030cca51b6f993f30040e1898b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.27099609375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_5cf8ecf7ffc4e50ccc33fd01836b5481(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2156982421875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_4de42851f0635cffc7ceb4b5fbf92113(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.383056640625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_9b9a4524d2b3a3fc86b9f863ea5768d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.01169586181640625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_a4529075a236c6543378307c06adc502(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0794677734375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_b23bdca55589b26193ecaaa0e949359b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.291015625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_95d4d5f0b8572a6d454bd8fa4f1bdc74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1552734375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_04ab921faf8efa3c2902cd566644cd57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.37744140625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_ba9d1360eaa15ea48aabdeceda65ffce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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
class TestPrimitiveOp_153492dbc634ccd79c65b8e6212d0ab8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.40234375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_67292aa1cb293a7de0e60a1c4c92ac67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.221435546875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_89409c0dc85bdc0a7607f2d03bcd3220(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.424072265625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_4433859c5644bfa80fb7df80c97ea089(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.310302734375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_076a3c0cd0b9618f267e3e4641b8c877(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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
class TestPrimitiveOp_03f03bd1ee3d1deb91f8e218aaafb45f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.487548828125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_d9a46895d76805bfed14ee4bf5953446(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1947021484375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_81c153bbbf7f3e35f04b3fef6b5ad334(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.18994140625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_29e4beebe9d8a391ee159b2f83c8f6c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.345703125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_1bf03c87b6ae249fd3a52343d7605154(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.359130859375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_b98b14482967baae04dcb2f9f45c0f46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.403076171875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_7a8e51bb7405d7671ec4e00328ee09e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.12548828125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_b3d5a94908dae9a29ea027ca2a0a4ff9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.03143310546875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_1451ded98678bb1830d725ca70b71be1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0244598388671875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_9ef3ae06d0f1945135fe7397b7eb8605(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.474365234375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_67f4d7d834c8387d7557ff518f6041a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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
class TestPrimitiveOp_6dcf637a84f3a22e5f9f6f3149649fcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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
class TestPrimitiveOp_fb7cc5f1d6c51f511b3712cbc4b8d7ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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
class TestPrimitiveOp_0af87172b67315bf4c8d1cf0429bdb1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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
class TestPrimitiveOp_53f2d4ec034b5bbcac53679fea8a4e24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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
class TestPrimitiveOp_cd73fa125661714e2fa88b740cfcade7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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
class TestPrimitiveOp_0d4639540405c5171158bed53686a9db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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
class TestPrimitiveOp_17bc7dd608eff3daae3feb173f7734a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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
class TestPrimitiveOp_c448c6b21046ae546b3d256eeb432b09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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
class TestPrimitiveOp_713a57cdf815593f6212759c37089de2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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
class TestPrimitiveOp_26576b831764798e20300d7cf084344c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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
class TestPrimitiveOp_7909d2e5ade4a9a8faf840baeb81a317(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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
class TestPrimitiveOp_a7552390dd22cafd85a230a35b69bd16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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
class TestPrimitiveOp_8d36596b415e77804ae42c9d17d9bf52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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
class TestPrimitiveOp_4c36c42a1922ce76b644bc72845b1c8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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

class PrimitiveOp_4501361c9f678872808361e7d5f525a4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_77a91341aae0c13fbff3c101e6ac9869(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4501361c9f678872808361e7d5f525a4
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
class TestPrimitiveOp_b8c64075c11c17124467979cb7d7cebd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4501361c9f678872808361e7d5f525a4
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
class TestPrimitiveOp_86b709ee1047ad38d7dabd1785883406(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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
class TestPrimitiveOp_5f601808e241f18007c710d649d54e74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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
class TestPrimitiveOp_3d0bf532aa326fc0d4bcb7c78f979f2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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
class TestPrimitiveOp_d876e4adb6ada162dd5ba0ba0514e6d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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
class TestPrimitiveOp_31187189ea7d8ab9b8e96136b05eaeb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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
class TestPrimitiveOp_d2c86ff5a371b0f85a4647b2883ddacd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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
class TestPrimitiveOp_5bdbb9a3ba6b838cbbaba38bf883b4f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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
class TestPrimitiveOp_85b9a00dabf4e6202dc41a9466425ccd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 800, 1344], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.2438969910144806]], [[0.37806886434555054]], [[0.023386424407362938]]]], dtype='float32').reshape([1, 3, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_777ada4e988d47974dd15581bf8e3c8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 100, 168], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4426266849040985], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2bc4c1c02a21fe676d9c81375d9783cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 50, 84], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.35133931040763855], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1f5795861128bbfdbd4500b50fd1ecc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 25, 42], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.21347853541374207], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9e7cd7e9619cf6b1eddae2a9c68ed6b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 13, 21], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.434526264667511], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_195ff03c5765c9437c9f432c81b0cd9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77f041bf03f78ba033c7c7b6a51e8d21
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 7, 11], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4117897152900696], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_23a8cccffbfadc370a9e40378225cc65(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d1c944348e97c9586acd21b87365a763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23a8cccffbfadc370a9e40378225cc65
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
class TestPrimitiveOp_b536c27385fb009770e96358b8309820(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23a8cccffbfadc370a9e40378225cc65
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
class TestPrimitiveOp_5b08ea37f8a21b042b2892e2c13eff5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23a8cccffbfadc370a9e40378225cc65
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
class TestPrimitiveOp_23e9e7c7d7cc75b10c5b439423f0ff46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23a8cccffbfadc370a9e40378225cc65
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
class TestPrimitiveOp_e83fa90cbb59bb7f09968e725a82dbf6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23a8cccffbfadc370a9e40378225cc65
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
class TestPrimitiveOp_a661fe3714e5af3dfd5f7702630b5a15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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
class TestPrimitiveOp_3f5c1d2d9bb0ea46ad6d6da2fb28ed06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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
class TestPrimitiveOp_3ce0c0d3e97e5c27ed6474174e54fcd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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
class TestPrimitiveOp_2104014fc4bfbe0666e634c566eb5c0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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
class TestPrimitiveOp_b1ddbac6d84a80d7dfa4ef4aeeb5d600(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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
class TestPrimitiveOp_14ff7021fb46c08590a85b240d172ebb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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

class PrimitiveOp_f8c3d412899464bfbc225eb9c55d561d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9afd68a3020b30e6e71b4fd9e64560ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8c3d412899464bfbc225eb9c55d561d
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
class TestPrimitiveOp_e6c9f09677f65badbe91a5409f4c41c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8c3d412899464bfbc225eb9c55d561d
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
class TestPrimitiveOp_f45255d46b9c0d42ffa4633a63005472(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23a8cccffbfadc370a9e40378225cc65
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

class PrimitiveOp_02cef8cae179a320c6d63d07c0cfb1df(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b2054ff88c4787d7201774850cb030d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02cef8cae179a320c6d63d07c0cfb1df
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
class TestPrimitiveOp_58fcb1f079631eaaaf3ece04bca88464(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02cef8cae179a320c6d63d07c0cfb1df
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
class TestPrimitiveOp_f7b74205f22f1043f22be2da636d55a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02cef8cae179a320c6d63d07c0cfb1df
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
class TestPrimitiveOp_a735be2a027c53946c14d01f351f409e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02cef8cae179a320c6d63d07c0cfb1df
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
class TestPrimitiveOp_752a991188fe33fcc389f695a1bb76b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02cef8cae179a320c6d63d07c0cfb1df
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
class TestPrimitiveOp_4a39b4a97aa4aeb0f830d97c6df6c907(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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
class TestPrimitiveOp_30434709f97c4a7a9761118046cd0b93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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
class TestPrimitiveOp_62776f4bb7c4d75dcae2a0ca06e7aff0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.054487742483615875], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_37768e09c2f955d1278fedf1233d1226(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.21072040498256683], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_cf8035b025e57961c2f2e8be6b6f5b0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3370343744754791], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_f21bda84a50513961e4997c84a223d76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.047739412635564804], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_0f97b3310138fc097561ce254fd5d694(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.15943484008312225], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_7010c53f679bfa035af4e4b8b228dfaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.45699337124824524], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_9e0325c9a60a2935379343874e947f4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10705216974020004], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_4ff97c6ba91999ee623b58d693dfbdf0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.15463751554489136], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_c32de840f49a4ad71caddeb8c57097e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3243096172809601], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_a8ec2aabf07a7161a21c37af70317276(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2100619077682495], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_5db8c8932a6413c5a3f698e709b5ab87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.30606868863105774], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_731fc8ba424d006ce31da262eddf2ad6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4733254909515381], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_6b89cd5f0527b426699fcc8b9da60dfd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.45036402344703674], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_95171ec61430539c1b5478a42459e7e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4187791049480438], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_fb1688c81ddc76c5fc86e23794f59ba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2859806716442108], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_9df905876696fcdbae0483ad37ca5333(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.49253273010253906], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_9cc936cacdf4f666cfb14a1a0f102f46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.29879528284072876], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_dfb9c0e9dcb7b73a9237753106b37009(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.28024914860725403], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_c0a2595455ead9f47f8284f562c4b63a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2171127051115036], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_d44888b1ae57383e22f5e22478052c41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.205111563205719], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_0f6395f1982afffa5ae8e2828de4ae39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3341596722602844], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_87b8390d5b241bed5d9dfe4e590720fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.28429579734802246], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_2d74336b3d0bbc1b272eaa883710e142(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.016756678000092506], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_dc9922fd698c1e103deff4dcfcbce2cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.26156091690063477], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_82aba71e49dce1c9826e796c896f9d16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.14741328358650208], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_923cb98f6548b7034d66cd0514d65f0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.21399670839309692], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_11ac604ca031ab025657d8b5473b79d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.40264075994491577], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_ab7de49224946b4e5ce3b03cad48aaf6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.02259201742708683], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_12ddf18703a25d6197d0d20a14128dc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.28167831897735596], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_20187196c330bd726c23a49bdb83b73d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2459322214126587], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_8ef4b86ec21cd2e38ccb9456a19c7165(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.42528459429740906], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_fd8cff5652069b93696d89d27876327d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2386317253112793], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_a728b79302b6e50d93651fd6049529bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1340242624282837], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_ad97bf5ba972b50397eca6e5e7256126(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.419644832611084], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_dd8feec9bd55774b32e1cd0dcbcba210(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2740958333015442], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_35094aac632bc16bbca2b4767b1ecd44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4016909897327423], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_ae6ead2f7f715d1938aa68bd1dc06355(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2894291877746582], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_bc1a9c5f9c2517b6aa6e5a14e0eb9292(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2736552059650421], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_aa47c5de95bb713e0372360f85cbe2e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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
class TestPrimitiveOp_25850b78d49573903fdbf385fa170d5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.228871151804924], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_387592df9b4823ae17d980b849969a5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1975477635860443], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_ac8aa62edfa64f80c8b8f1aba9195fba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3624515235424042], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_487a14439d298b5aa6604af596158f90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4675005376338959], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_2c923b2f8319ddb0518562f7097434e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1207498237490654], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_a180e40bb0d1ff4700e45c23297275cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.05389200150966644], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_112f2040047021c4f5127e94e5d328c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.28303852677345276], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_2bbfb6294444041ad68b55ebafdd640b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.23547503352165222], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_6755c037318c0dcae11cb1f9960aa831(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.44396933913230896], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_7b3a5806b3a719c0797b3e3f020d5580(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.39385876059532166], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_ef6c7acc63de5f104cec5fe65bc64f67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2808399200439453], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_3cc57c2b13ec0b4827a54d0883c139d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.011308269575238228], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_ff324fd1dfd8c66e7a479e141c3ce9ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.44586485624313354], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_91b34f9dfc6a479d47ccb9f90d66795a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_349273e9f2ee736aa41a32d44d12bbbc
    def get_inputs(self):
        return [
            paddle.to_tensor([0.48435136675834656], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_382b659ffaa8149972086b9c4cdd1ba0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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
class TestPrimitiveOp_4c2b7adb1d9c6933b77ca57ad501834b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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
class TestPrimitiveOp_449a5713b8fd221aefb205bf88aff366(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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
class TestPrimitiveOp_98bf105edd3694dc2de341d26e5ed064(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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
class TestPrimitiveOp_5a8246cb5c0fbd762619e750b8c99292(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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
class TestPrimitiveOp_31629077a8d3f9b543616eb575da9939(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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
class TestPrimitiveOp_40b0f8ae208262127bc9fa2ad4d86ff7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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
class TestPrimitiveOp_868021a682c4f93a49dff8595f198503(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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
class TestPrimitiveOp_d240633714cb7bd35024c2dc6a04aea5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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
class TestPrimitiveOp_e744de92ca039a12bdf3bd7e12d91a40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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
class TestPrimitiveOp_903e2702aceb1b352882941ec74257ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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
class TestPrimitiveOp_967581aa096d70e477c1dd09488b0002(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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
class TestPrimitiveOp_b8e16de42ba25e342e3b2fe36b270237(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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
class TestPrimitiveOp_855c21cdf56a1531dc8ac2af8cc972fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10919189453125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_79d378741548fadcb8027b0bb3afa70f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.025909423828125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_ec4a81fc2c83c2140a74973e5cbb4d9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.299560546875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_9e5e2c6b50cd8107e2b4483cbd6e624d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.317626953125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_2789ee9400c08774afaf6547692f740d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.25146484375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_bfb003648d8eace4875ad6be41f6b8c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.25927734375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_467c8784606f321992ded70ac09f5369(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.45263671875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_ca7bfc2f13fde2dde26bc7b064102a89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.28955078125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_07877f4fb68f208ca7ed4604a5db5aac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10296630859375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_5f050d9e4c289fc3b0230271cd0d5177(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.293701171875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_afd5bcd5c9361fc344e2cb80d3c03aba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1744384765625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_4e86dd49300fe664c271fce4da7aea77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.08538818359375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_871e08a8407eb72ec2e17ee119f73f29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.40185546875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_61ccaec0877e3fc62f65e7cf137efeca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10150146484375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_5cfb2336adcf05c9f498cccf409a29ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3974609375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_f83aded8bd1a3be5c70e187f53b3cb1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1925048828125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_e7c6e2a8af6080e315492edcf509d614(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.391845703125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_0b0dd615dd2d3042a29220da6dc7eb57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2890625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_996f94bec27f264b6678a81a6c859b75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.466552734375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_f35cd31cb1d274fb3a668e34cbdf1a1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10125732421875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_1f3c6b456428076ad49a395dacb3a029(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.08502197265625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_11adb0d759409c80c8723e6ae0465c11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.363037109375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_bf7e462252d4adb145a25c9769d141d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.28173828125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_00bcf5b08a392b4c803ebe51f52cc609(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1937255859375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_ca19bb20f42e4952f14a2a10fcb321ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1319580078125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_1a58b719dba26da8522a08925ff45af1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.281494140625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_4a37b45e63f0340a007918ea19ec6221(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.476318359375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_b2267252d690e5c8de8564bbb4f7d1e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.01165008544921875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_098225a28861e5220e504521d32e4dfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.212158203125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_b90f2458b1f11d0e7a6e7e02a0d46843(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.380615234375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_c29165f247ee146b732865e2052f1f58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1824951171875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_3eb21b66c513fc178a49b7d22d73d04b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2254638671875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_54a9374a988fae7026ee6df9f7eb86db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0010156631469726562], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_060ceae6882a88d2a6cf502bd9c4b6b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1300048828125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_d596ad5632b1582f249c70a9908f4859(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.196044921875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_b56484a24f5517b1b2c723e20e5e1531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3515625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_c6fb506f7393ed552b52bac27af48756(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2130126953125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_5ce61f0844460b72ebf985f76bfba7de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.26806640625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_fb70d1ee89b6b75203f0461d8aa7ad81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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
class TestPrimitiveOp_7af8525e62824125895e70cfbf330c3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1673583984375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_cf49ba55741e1ba9e41bcebfdff41d19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1259765625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_27dd904b4d597d8eae8a90a351013d6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.421875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_8cfb56376e12a3dd66ec7d10c9ca0101(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.487060546875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_648ba190f4b02652b257d9805df69e6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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
class TestPrimitiveOp_eb7058d6120078d4304c28cad205ccbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.241943359375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_da545d008043f405e285de7fd079de96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0538330078125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_2cee2d9c0996de8c0246a346cb55ccb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.219482421875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_ffff2843729eb28f20f6eb42c5dfe039(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.369873046875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_964b3a5ddb7826b6d69f4604c7355a8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.499267578125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_7fbdb45f05e75a327f642d3cc603ce79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.314453125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_fe40623e8f6cc72207d089ff4e643b3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.36279296875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_35130ecebf521521b259053ff1149288(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.161376953125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_ef9d7e81f147ff06279f24ba27ed88d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.09332275390625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_2e67e1ddf1929558c0da019ad025a48f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba8b969029fc634ac79fa63a1ec1ab6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.070556640625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_b9f4e6729f3718ce82927f4760741468(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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
class TestPrimitiveOp_a1343665006b1c1d33d7020e2207f874(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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
class TestPrimitiveOp_43523b80e855a9cd4360bbec12a899d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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
class TestPrimitiveOp_c007c5859803cee43ac88d68e17213d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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
class TestPrimitiveOp_6cd2ace211d8ac7472d17f7a69001614(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 20, 20], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([[[[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]]]], dtype='float16').reshape([1, 24, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_16dd648abcfad64c0ff4128b28db0741(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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
class TestPrimitiveOp_4e63d1238514de87988db2a96d10a665(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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
class TestPrimitiveOp_dd6382db91c7fa9f1515f9d2434f5f6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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

class PrimitiveOp_61c00e6d18bd32f6405b333aaeb1ca95(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0e503a87f95e64058515543d09e40e9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61c00e6d18bd32f6405b333aaeb1ca95
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
class TestPrimitiveOp_1ecf4527f5db1375ff80041579b04e00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61c00e6d18bd32f6405b333aaeb1ca95
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
class TestPrimitiveOp_cab551df195d03b8b1a0cb400f84a4f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61c00e6d18bd32f6405b333aaeb1ca95
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
class TestPrimitiveOp_0bf000bece2733b711edd19da38bfa2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61c00e6d18bd32f6405b333aaeb1ca95
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
class TestPrimitiveOp_ced7658726b1a39e8b5cce57d6572d0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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
class TestPrimitiveOp_9cebf72abe90170f09bfee3372f16e93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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
class TestPrimitiveOp_4cff2bc187ea41e5faba82f14ca32b03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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
class TestPrimitiveOp_21864738520755cfdd23516f62d9925d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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
class TestPrimitiveOp_86233b93fa7296445b008d8f38ad4855(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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
class TestPrimitiveOp_3919d87949d4a9d7a9ebc0ae9fc21f58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02b9d20c36611ef2fad29beb0243d068
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
class TestPrimitiveOp_fb4d27e256d038bccfca8ae726bcfd2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90135cce2357169f6db23af094f48dab
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

class PrimitiveOp_0f9b472f23029c987d305a365f776285(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_546a852fa1f3d230e6928739e8aadbee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9b472f23029c987d305a365f776285
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
            paddle.to_tensor([160], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_90d5139045e779ecece9828cc428ef4e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0f159d938de332d33d9ae23e0d51e3ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90d5139045e779ecece9828cc428ef4e
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
class TestPrimitiveOp_b9a4074bc6262bc9231a52eedd1aa34b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90d5139045e779ecece9828cc428ef4e
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

class PrimitiveOp_b5376e4eb5f69cabfddc3afa798ca875(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 26, 512], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 26, 512], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b030548f2383713b6dd37167eef183bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b5376e4eb5f69cabfddc3afa798ca875
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

class PrimitiveOp_25a87951109d1a21d9596a75a3e62cba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 256, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0101528d0ff5678b5383200fdd99b2c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25a87951109d1a21d9596a75a3e62cba
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

class PrimitiveOp_85cd9990436f2ae8813ea5ba3426ff22(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 512, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_44bbde6ffc187616fbf391d5bcf0fb2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85cd9990436f2ae8813ea5ba3426ff22
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

class PrimitiveOp_86bf59886185f0e6cd7c42e863a0c521(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ac230219b801746e9629888ab0481240(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86bf59886185f0e6cd7c42e863a0c521
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
class TestPrimitiveOp_7cd28e754be275bd1cc8b7886ae385a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86bf59886185f0e6cd7c42e863a0c521
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
class TestPrimitiveOp_22c5b1c550c55957a89acd8f6271ac24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f9b472f23029c987d305a365f776285
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
            paddle.to_tensor([240], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c13876966c9e62bb2b8d3893967019ec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 40, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 40, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_249580c20784761554631f537aceb681(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c13876966c9e62bb2b8d3893967019ec
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

class PrimitiveOp_f47f8cde93bde2463ad8c9966f0858bf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 64, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7a7552ad7bb9109cd9a9150e789b4089(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f47f8cde93bde2463ad8c9966f0858bf
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

class PrimitiveOp_07bf753e32b01b76520d7697121f07ad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 240, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b00cbd53984d790cdb2ffcbfc853a33d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07bf753e32b01b76520d7697121f07ad
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

class PrimitiveOp_bb7533bfe2ab31d32437f21c72cd99f3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 336, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 336, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d97a5f455cb62eb81732749cf54e9d41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb7533bfe2ab31d32437f21c72cd99f3
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
class TestPrimitiveOp_5cec48b2113e728040053333870f00c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb7533bfe2ab31d32437f21c72cd99f3
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

class PrimitiveOp_af8a58da019b9911d1523aab76aa63cd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 480, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 480, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ebd20a8b810fc103ce09595da3a62a42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af8a58da019b9911d1523aab76aa63cd
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

class PrimitiveOp_426962b52f0a57ea2b23fd0d28b494d8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 384, 20, 20], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 384, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d00ff8c898e0b82b5744af98802675a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_426962b52f0a57ea2b23fd0d28b494d8
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

class PrimitiveOp_6df2767854a72be3751a119d85b3dba0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, 40, 40], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 192, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e7a93390d495f32f4eaa79d9650ab93a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6df2767854a72be3751a119d85b3dba0
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

class PrimitiveOp_772579f9d453d4390c9f7c7db513d277(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 80, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6d21137104a50597f71fc57492b1c568(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_772579f9d453d4390c9f7c7db513d277
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

class PrimitiveOp_aa4c1531bb5b5beae8f7e498bb997d2c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[1, 3, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_08b12ef56bbc72bccf4f9b32af26d811(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa4c1531bb5b5beae8f7e498bb997d2c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 800, 1344], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([[[[0.2685546875]], [[0.08740234375]], [[0.287353515625]]]], dtype='float16').reshape([1, 3, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_87c9e86a7589c21cce4fc78a49f36cae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a01873d7f10d0367dd9ad5774600a557(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87c9e86a7589c21cce4fc78a49f36cae
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 100, 168], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.311279296875], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_638ca32d5a10e138ef6fe2e691322d55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87c9e86a7589c21cce4fc78a49f36cae
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 50, 84], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.482177734375], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7ce4154f54c27709a89a5d0baf20b9f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87c9e86a7589c21cce4fc78a49f36cae
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 25, 42], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.303955078125], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d91f25c347e24faf4a03a38e328742b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87c9e86a7589c21cce4fc78a49f36cae
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 13, 21], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.14013671875], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_387115da1218c5b3f42fc2d9678c34d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87c9e86a7589c21cce4fc78a49f36cae
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 7, 11], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.03875732421875], dtype='float16').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2799c8637e7f6ba2968298ad8f0873a7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 80], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c21b04336927fd8da4c0dfef873f9361(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2799c8637e7f6ba2968298ad8f0873a7
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
class TestPrimitiveOp_c4b6bb7fc587c3cf84eb5836b61c103e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2799c8637e7f6ba2968298ad8f0873a7
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
class TestPrimitiveOp_f06c5fc08f4399e0ea8bce8027cdd9c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2799c8637e7f6ba2968298ad8f0873a7
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
class TestPrimitiveOp_24aebc4055c29c2fdb08f834bc99bfdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2799c8637e7f6ba2968298ad8f0873a7
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
class TestPrimitiveOp_a64a0debb1dca3bef03520842ddd8152(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2799c8637e7f6ba2968298ad8f0873a7
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

class PrimitiveOp_421705b9e03712607abc1c5ba4c63d0b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 72, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 72, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_62670d31e162c1086a09cfb267e7ef10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_421705b9e03712607abc1c5ba4c63d0b
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

class PrimitiveOp_cda3ecbdf132dfc2078f2950265ec2bd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 120, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 120, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_16b8c391f146f1c313f70cc537c9f966(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cda3ecbdf132dfc2078f2950265ec2bd
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

class PrimitiveOp_af809f60aaa86de5f779f4a04db52410(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 480, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 480, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_406733f9da3a4411178f8fc03c5da716(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af809f60aaa86de5f779f4a04db52410
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

class PrimitiveOp_d9c9136a77722eb7c9960e20a1908725(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 672, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 672, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e5ae904533e86bf1565b8ce726e2dc00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9c9136a77722eb7c9960e20a1908725
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
class TestPrimitiveOp_a3a4d4b6d1f81e5e2a113b32a1edd735(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9c9136a77722eb7c9960e20a1908725
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

class PrimitiveOp_c77f0675f4a5da288cb29a4f99fd975f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 960, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 960, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bef21f04f1a77f2f9dfb11441e3ba32a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c77f0675f4a5da288cb29a4f99fd975f
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

class PrimitiveOp_30f434fa3b27fca426dedbe9efa81a40(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2, 1, 9, 112, 112], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 2, 16, 9, 112, 112], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3c3fdd1120d9895627a97f11e52a7983(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30f434fa3b27fca426dedbe9efa81a40
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

class PrimitiveOp_1113cb7117df927de08f45c18f2ec145(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 1, 49, 56, 56], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 4, 16, 49, 56, 56], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c174f0f96c69af141cf0f118c0824563(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1113cb7117df927de08f45c18f2ec145
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

class PrimitiveOp_ce360438407deeffab147e462f4dcaa7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, 1, 49, 28, 28], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 8, 16, 49, 28, 28], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3183549bc3997dbeb28624dc5a0e6520(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce360438407deeffab147e462f4dcaa7
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

class PrimitiveOp_c69be721a6ce5b910cefc1ec4d64ad4f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 1, 49, 14, 14], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 16, 16, 49, 14, 14], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dda40788e34dfb5b5528cbdc592f004e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c69be721a6ce5b910cefc1ec4d64ad4f
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

class PrimitiveOp_3570de89ee43659f96f5b8a6bb92d057(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 1, 49, 7, 7], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 32, 16, 49, 7, 7], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_11e11f197ad57c3983a07b8306facb93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3570de89ee43659f96f5b8a6bb92d057
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

class PrimitiveOp_5d2693ec05df3906d17f8a6606a188e5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 16, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b686b90620bec1900d73fe5a08de2363(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d2693ec05df3906d17f8a6606a188e5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.034269895404577255], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_c415093816a6afc60d8a5721ee842d43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d2693ec05df3906d17f8a6606a188e5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3553077280521393], dtype='float32').reshape([1]),
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

class PrimitiveOp_f6ce0805f387f39d16e1535160edb408(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f4e8039e5ae2c99151b22d5ba844ff3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6ce0805f387f39d16e1535160edb408
    def get_inputs(self):
        return [
            paddle.to_tensor([0.45771870017051697], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_545fb7e388f08f1e750cc66da61b89ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6ce0805f387f39d16e1535160edb408
    def get_inputs(self):
        return [
            paddle.to_tensor([0.025557570159435272], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_7bfe25b397966332eabd9576adc1ce9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6ce0805f387f39d16e1535160edb408
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4053143262863159], dtype='float32').reshape([1]),
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

class PrimitiveOp_eb5dc535d535efb2f4d64d27da775830(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_291cba942ee6218e620552b5bbf36ea4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb5dc535d535efb2f4d64d27da775830
    def get_inputs(self):
        return [
            paddle.to_tensor([0.13564513623714447], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_2ac0ba8b836993e1562b1d4b7c6c5ed7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb5dc535d535efb2f4d64d27da775830
    def get_inputs(self):
        return [
            paddle.to_tensor([0.01814817450940609], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_506fc3d25710b72a0d8c13a5743a9bba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb5dc535d535efb2f4d64d27da775830
    def get_inputs(self):
        return [
            paddle.to_tensor([0.21919342875480652], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_e49a9845f5c38c9e3da4e31b717e00d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb5dc535d535efb2f4d64d27da775830
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2764962315559387], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_99914cbd42c63de0eb143039b75b1fb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb5dc535d535efb2f4d64d27da775830
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2401210069656372], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_9d04d9d7e2af6ea6eeec8a5d6635a3a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb5dc535d535efb2f4d64d27da775830
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3128359317779541], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_f89f346a8245fdb65a19abce6147b21b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb5dc535d535efb2f4d64d27da775830
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0034283464774489403], dtype='float32').reshape([1]),
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

class PrimitiveOp_a6af68e33730ac05b49bc920ca5d6efe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_582858e3cd55c6b7b2430f4800d7912f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6af68e33730ac05b49bc920ca5d6efe
    def get_inputs(self):
        return [
            paddle.to_tensor([0.11662230640649796], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_2b1cd312b6b33d2a8735ccc7270115e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6af68e33730ac05b49bc920ca5d6efe
    def get_inputs(self):
        return [
            paddle.to_tensor([0.47155091166496277], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_6dcf5b8225f3a64605a90054bdf2b784(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6af68e33730ac05b49bc920ca5d6efe
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2890031635761261], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_c308854cac4a5b8f27ea9c577237abe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6af68e33730ac05b49bc920ca5d6efe
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3942105770111084], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_7f8473bc1130902f64dc8be39e9d68c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6af68e33730ac05b49bc920ca5d6efe
    def get_inputs(self):
        return [
            paddle.to_tensor([0.25721466541290283], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_5b2b87acd002cdb99043ae8d2f7222dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6af68e33730ac05b49bc920ca5d6efe
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2686125636100769], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_0bc5892ee14638898de7c59d60827168(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6af68e33730ac05b49bc920ca5d6efe
    def get_inputs(self):
        return [
            paddle.to_tensor([0.23588304221630096], dtype='float32').reshape([1]),
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

class PrimitiveOp_e5d420d811f3ab50968af0e00a0906eb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a02ee2d27c5f6ebba3dd8cb78660deb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5d420d811f3ab50968af0e00a0906eb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.07459518313407898], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_c6137b164804cd425d7e9f8f9915e82f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5d420d811f3ab50968af0e00a0906eb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.07328919321298599], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_a5cd1a6f8345a47095ba8d6e807e73da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5d420d811f3ab50968af0e00a0906eb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1305624097585678], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_b3632692266a7a73ef92467ae7e32ee0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5d420d811f3ab50968af0e00a0906eb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.12345968931913376], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_a0548c33490ebc8c29a8294a7540bd25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5d420d811f3ab50968af0e00a0906eb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.32259705662727356], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_c660a576cba7f73c7a34616f12238fcf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5d420d811f3ab50968af0e00a0906eb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2142874300479889], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_4846262b7e6c7e9eb7dd627de9159e45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5d420d811f3ab50968af0e00a0906eb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.033593520522117615], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_28187d2329361473d5093b75ca860437(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5d420d811f3ab50968af0e00a0906eb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.33284541964530945], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_a22e80b52e1b98aa3dc2b436425619dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5d420d811f3ab50968af0e00a0906eb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.09706348925828934], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_e9583d148cc99e8858bbbd8773603576(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5d420d811f3ab50968af0e00a0906eb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.16094255447387695], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_4bc4c916b31f9252cf7ae87dbfcd4a99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5d420d811f3ab50968af0e00a0906eb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4290601909160614], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_cf50a33b514469174d8d69986aa854b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5d420d811f3ab50968af0e00a0906eb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.055179841816425323], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_fb7cacabd38411338d928c4ccd68c859(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5d420d811f3ab50968af0e00a0906eb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.042275089770555496], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_776eb7fb76b1bbf60d136feb3325d22f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5d420d811f3ab50968af0e00a0906eb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.061161477118730545], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_06577e1928ba18eadff15b4e8f9bd12f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5d420d811f3ab50968af0e00a0906eb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3398377597332001], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_e63bb80138236f09869856ed5b11bc5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5d420d811f3ab50968af0e00a0906eb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3320675194263458], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_da9d556ec2f68cfe9811deec18b2a8e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5d420d811f3ab50968af0e00a0906eb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.44745224714279175], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_030eed037d1568bf5e572f92706b3532(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5d420d811f3ab50968af0e00a0906eb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3121994137763977], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_358d78cd1a2ef19fceebb001f8158a6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5d420d811f3ab50968af0e00a0906eb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10801368206739426], dtype='float32').reshape([1]),
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

class PrimitiveOp_3d5112d9c946203be8f957691d36cce8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 192, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cb1289aa2dc645f78ee36a13f2917a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d5112d9c946203be8f957691d36cce8
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

class PrimitiveOp_3facc485d59a84ad63c81dd8401dcc51(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 384, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6044179b421a5635fc109371a1b24392(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3facc485d59a84ad63c81dd8401dcc51
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3038066625595093], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_f8f252433d0dd9ab432b33091144f6f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3facc485d59a84ad63c81dd8401dcc51
    def get_inputs(self):
        return [
            paddle.to_tensor([0.23026257753372192], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_fb8fed71efda15f04e5c4ccf287d01b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3facc485d59a84ad63c81dd8401dcc51
    def get_inputs(self):
        return [
            paddle.to_tensor([0.09069259464740753], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_67d4e3e1685f7bbe7e16ffdf07df3ad1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3facc485d59a84ad63c81dd8401dcc51
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3072358965873718], dtype='float32').reshape([1]),
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

class PrimitiveOp_843c795c904b4dd89d0f9575184e65d2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 384, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 384, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9bac50ffd8119129861f0992432ed898(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_843c795c904b4dd89d0f9575184e65d2
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
class TestPrimitiveOp_76ce57723d8bb9e25d553fe433470ce5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3facc485d59a84ad63c81dd8401dcc51
    def get_inputs(self):
        return [
            paddle.to_tensor([0.17319104075431824], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_7546697c1e39f21e71819fcf7049ee7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3facc485d59a84ad63c81dd8401dcc51
    def get_inputs(self):
        return [
            paddle.to_tensor([0.014830023981630802], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_d53227b2244908c8bd1f84ae3c13e394(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3facc485d59a84ad63c81dd8401dcc51
    def get_inputs(self):
        return [
            paddle.to_tensor([0.24390298128128052], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_46a8b9eccd3f0c77b5664c100bf988a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3facc485d59a84ad63c81dd8401dcc51
    def get_inputs(self):
        return [
            paddle.to_tensor([0.25205427408218384], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_5b103d336faaf4bbb6ffa5cab636437d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3facc485d59a84ad63c81dd8401dcc51
    def get_inputs(self):
        return [
            paddle.to_tensor([0.19475553929805756], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_1304fe2fdfadf42d3928697c37b48ce3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3facc485d59a84ad63c81dd8401dcc51
    def get_inputs(self):
        return [
            paddle.to_tensor([0.21722935140132904], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_22d399661ad3d9d336ffaf90b1c0d062(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3facc485d59a84ad63c81dd8401dcc51
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3297799527645111], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_3af4efc03438fa8867f0686ed9386d1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3facc485d59a84ad63c81dd8401dcc51
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4012543857097626], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_98c1e0753a5d1ee312d03b8d66d74586(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3facc485d59a84ad63c81dd8401dcc51
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2193007469177246], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_bbd5493a3797b926e7bf48d4c6864881(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3facc485d59a84ad63c81dd8401dcc51
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3298938572406769], dtype='float32').reshape([1]),
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

class PrimitiveOp_685215e39df696ab66134852db25f88e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_90d8b29dc44eed0235270f83dc573305(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_685215e39df696ab66134852db25f88e
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
class TestPrimitiveOp_d90904a54ab2c6a7e8da0236ef5021a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_685215e39df696ab66134852db25f88e
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
class TestPrimitiveOp_17125c0688756e286c175a5215c901e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_685215e39df696ab66134852db25f88e
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
class TestPrimitiveOp_bdbd840419e39207e64d84f1f45040d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_685215e39df696ab66134852db25f88e
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

class PrimitiveOp_a2cf6a2f6cd41e801025143d99d97402(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 24, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 24, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b481a13cf43fdcd065f62eb75292e15e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2cf6a2f6cd41e801025143d99d97402
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
class TestPrimitiveOp_a6202b93208553920d92d00086c366fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2cf6a2f6cd41e801025143d99d97402
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
class TestPrimitiveOp_e91b06369fd2e4e3038408a5135dad0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2cf6a2f6cd41e801025143d99d97402
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
class TestPrimitiveOp_9d0c27dc97157c8624a09f5e0bf0a1c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2cf6a2f6cd41e801025143d99d97402
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

class PrimitiveOp_4c580c6346438e5cde81e8640578a023(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[32, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 32, 56, 56], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_37f9d5ca22684623c62091cc901550fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c580c6346438e5cde81e8640578a023
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

class PrimitiveOp_811303f4f5679aba0801f50c4b499b00(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[64, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 64, 28, 28], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f164394ceef71f9d2fe25a121fecc92f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_811303f4f5679aba0801f50c4b499b00
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

class PrimitiveOp_93402350b014df242d3a2f8b9a0f32ff(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[160, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 160, 14, 14], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f47e344acdc143395af2db2ae6b01917(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93402350b014df242d3a2f8b9a0f32ff
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

class PrimitiveOp_e0be2bf345c3b75e087edcd86c1506fb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[256, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 256, 7, 7], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4d091bf78dd178239baad811c82dee78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0be2bf345c3b75e087edcd86c1506fb
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

class PrimitiveOp_c5fa5e543697a53fd72eb788888866f1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 28, 28], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 32, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4858ddf49860405ca35b257ddc6dd89a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5fa5e543697a53fd72eb788888866f1
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

class PrimitiveOp_5a1c5a5a76572602bbe3db9778fb4e4b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, 14, 14], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 48, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_01d18f40d2bf4b64b69d4f87aacdb152(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a1c5a5a76572602bbe3db9778fb4e4b
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

class PrimitiveOp_100cc4f0b79d3a96767b59137ad8a16b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 14, 14], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 64, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6774be7baeaaeeb1a23ef9ac423b8e56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_100cc4f0b79d3a96767b59137ad8a16b
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

class PrimitiveOp_d5b407b6d8d0dfecf4642f927b1a7828(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 14, 14], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 128, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f76e8896ce46b02366606fb47f7772a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5b407b6d8d0dfecf4642f927b1a7828
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

class PrimitiveOp_fad5435eb3868ed554e45a5fe3c08e5f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 7, 7], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 128, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_54239f0f352c19e34a8eb626ad57d3e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fad5435eb3868ed554e45a5fe3c08e5f
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

class PrimitiveOp_92523edf6ab38aeec4a049ccb4d9057e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 7, 7], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 256, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b03d56780c9b985517850817382810c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92523edf6ab38aeec4a049ccb4d9057e
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

class PrimitiveOp_f67b0a595cb0c360323bb1876232fd34(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 16, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c17952f5ab10246f85a355bb2b4b75de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f67b0a595cb0c360323bb1876232fd34
    def get_inputs(self):
        return [
            paddle.to_tensor([0.156005859375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_d07bc8b59926290832e53394b9094fb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f67b0a595cb0c360323bb1876232fd34
    def get_inputs(self):
        return [
            paddle.to_tensor([0.423583984375], dtype='float16').reshape([1]),
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

class PrimitiveOp_df4bd3f2745f3c372cf78d808f434ed0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d7b866831bb6d2b15ed257f91c9ef6ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df4bd3f2745f3c372cf78d808f434ed0
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1646728515625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_e89d3b30a02dd751bcb996dd6fd9a71b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df4bd3f2745f3c372cf78d808f434ed0
    def get_inputs(self):
        return [
            paddle.to_tensor([0.386962890625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_5463b0a75c61c760009891ef6bf17bac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df4bd3f2745f3c372cf78d808f434ed0
    def get_inputs(self):
        return [
            paddle.to_tensor([0.33349609375], dtype='float16').reshape([1]),
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

class PrimitiveOp_76f3d06fd684035c4e26fc5c7d6ca147(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cf4b62eded5557988cd28dec466d91f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76f3d06fd684035c4e26fc5c7d6ca147
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2442626953125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_4a258adfee3c34a07099d5359aa5a7b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76f3d06fd684035c4e26fc5c7d6ca147
    def get_inputs(self):
        return [
            paddle.to_tensor([0.26318359375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_80af0a9d94df50a3197a11c25fd2b8ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76f3d06fd684035c4e26fc5c7d6ca147
    def get_inputs(self):
        return [
            paddle.to_tensor([0.04327392578125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_f3079c62b7a5eba7b9c3b0bb1adf02c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76f3d06fd684035c4e26fc5c7d6ca147
    def get_inputs(self):
        return [
            paddle.to_tensor([0.12042236328125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_3005a7e94c3dc7f90fc9c5f48fda77c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76f3d06fd684035c4e26fc5c7d6ca147
    def get_inputs(self):
        return [
            paddle.to_tensor([0.202392578125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_14913dae1107695511fdb3ee951088f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76f3d06fd684035c4e26fc5c7d6ca147
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1036376953125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_7bdee186b473bd40a3fb5118f7f2dd45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76f3d06fd684035c4e26fc5c7d6ca147
    def get_inputs(self):
        return [
            paddle.to_tensor([0.134521484375], dtype='float16').reshape([1]),
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

class PrimitiveOp_5c0907bc0f8545b9b41bfefa0339dc28(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ddfa327a324df843641c668e7d4c68ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c0907bc0f8545b9b41bfefa0339dc28
    def get_inputs(self):
        return [
            paddle.to_tensor([0.37158203125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_7d140487359c7dc36e0c47b6ee233bad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c0907bc0f8545b9b41bfefa0339dc28
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0194549560546875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_b28d085e28a2f3fab4cc461682cd7484(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c0907bc0f8545b9b41bfefa0339dc28
    def get_inputs(self):
        return [
            paddle.to_tensor([0.12030029296875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_3a49f87c6566fe90250a44f899931f07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c0907bc0f8545b9b41bfefa0339dc28
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3505859375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_e219e4056a0eafa48b758b78b93c7c31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c0907bc0f8545b9b41bfefa0339dc28
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1939697265625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_563afba522e2b7301ee87f66d5436d85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c0907bc0f8545b9b41bfefa0339dc28
    def get_inputs(self):
        return [
            paddle.to_tensor([0.312255859375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_a48c5fe32953c10ec25d7566c0148705(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c0907bc0f8545b9b41bfefa0339dc28
    def get_inputs(self):
        return [
            paddle.to_tensor([0.020751953125], dtype='float16').reshape([1]),
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

class PrimitiveOp_24bf4640dbd46b3eeb546b06869acb92(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4802f6daa02d667cfa482f33c51ac06f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24bf4640dbd46b3eeb546b06869acb92
    def get_inputs(self):
        return [
            paddle.to_tensor([0.41552734375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_b1d68e45dcef39c3e0fabbaa67bd554d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24bf4640dbd46b3eeb546b06869acb92
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2044677734375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_413466d716617ef00ac992e27a3cc42f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24bf4640dbd46b3eeb546b06869acb92
    def get_inputs(self):
        return [
            paddle.to_tensor([0.47998046875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_3b8674dc8156d3ce3c9f252a66a44047(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24bf4640dbd46b3eeb546b06869acb92
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2447509765625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_a0123adcad415513deec711dd41d3258(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24bf4640dbd46b3eeb546b06869acb92
    def get_inputs(self):
        return [
            paddle.to_tensor([0.23974609375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_ed8c90b1fabe4e06e85da1aeeb106f6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24bf4640dbd46b3eeb546b06869acb92
    def get_inputs(self):
        return [
            paddle.to_tensor([0.21630859375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_be94f950e9f5d049a1d1c172cd07c7ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24bf4640dbd46b3eeb546b06869acb92
    def get_inputs(self):
        return [
            paddle.to_tensor([0.213623046875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_3ed565d67aa381112d0d52e93e7cf107(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24bf4640dbd46b3eeb546b06869acb92
    def get_inputs(self):
        return [
            paddle.to_tensor([0.36865234375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_4d196fba899249c23403ff068e92dd08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24bf4640dbd46b3eeb546b06869acb92
    def get_inputs(self):
        return [
            paddle.to_tensor([0.39111328125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_8a0bc94f04d15f423218d20996c76485(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24bf4640dbd46b3eeb546b06869acb92
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2152099609375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_5f099a86c68241ffa9f742a00c5afbef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24bf4640dbd46b3eeb546b06869acb92
    def get_inputs(self):
        return [
            paddle.to_tensor([0.25], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_1544b6410df11192438cea6efb9babdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24bf4640dbd46b3eeb546b06869acb92
    def get_inputs(self):
        return [
            paddle.to_tensor([0.27099609375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_1e698b66320d2a815a9e1c5800976f0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24bf4640dbd46b3eeb546b06869acb92
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2156982421875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_25264421bf89b60cb5db47776f756cd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24bf4640dbd46b3eeb546b06869acb92
    def get_inputs(self):
        return [
            paddle.to_tensor([0.383056640625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_ca71d67482f864f49cb43b53d31339bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24bf4640dbd46b3eeb546b06869acb92
    def get_inputs(self):
        return [
            paddle.to_tensor([0.01169586181640625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_9812a2766c4e8b9fdd1e276f4da6244e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24bf4640dbd46b3eeb546b06869acb92
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0794677734375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_ed5e77918870265d30bbde69873309ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24bf4640dbd46b3eeb546b06869acb92
    def get_inputs(self):
        return [
            paddle.to_tensor([0.291015625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_44bdd1733e7c9f3075c97781ab7c1745(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24bf4640dbd46b3eeb546b06869acb92
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1552734375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_f951ec619f9aa7c89c1294d43ec4774a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24bf4640dbd46b3eeb546b06869acb92
    def get_inputs(self):
        return [
            paddle.to_tensor([0.37744140625], dtype='float16').reshape([1]),
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

class PrimitiveOp_90d90adf304dfd9c18a5d81d20de13bd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 192, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_96aa12fe4a53e8e956acffdf47c79033(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90d90adf304dfd9c18a5d81d20de13bd
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

class PrimitiveOp_d1f9768dbc6ef556427844a92f14dca9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 384, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_407e9bc4af6e1e9ab65ff47496cbf37b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1f9768dbc6ef556427844a92f14dca9
    def get_inputs(self):
        return [
            paddle.to_tensor([0.40234375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_99eb3dbfbaa9d46275923827973fb30b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1f9768dbc6ef556427844a92f14dca9
    def get_inputs(self):
        return [
            paddle.to_tensor([0.221435546875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_80653141e9c1f23b2ed1eef0480234b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1f9768dbc6ef556427844a92f14dca9
    def get_inputs(self):
        return [
            paddle.to_tensor([0.424072265625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_5fb21f4d0cf072704cc383214f11f31a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1f9768dbc6ef556427844a92f14dca9
    def get_inputs(self):
        return [
            paddle.to_tensor([0.310302734375], dtype='float16').reshape([1]),
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

class PrimitiveOp_4262ebfd178c59cfbc57a0194b518a95(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 384, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 384, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_789041060cbb8f6676e13f34fedaadef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4262ebfd178c59cfbc57a0194b518a95
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
class TestPrimitiveOp_543d369e352cee239c518996aea128f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1f9768dbc6ef556427844a92f14dca9
    def get_inputs(self):
        return [
            paddle.to_tensor([0.487548828125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_b21756d8091244666d15ec490a75fe63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1f9768dbc6ef556427844a92f14dca9
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1947021484375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_7689ff410505e993016dbbc186462f2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1f9768dbc6ef556427844a92f14dca9
    def get_inputs(self):
        return [
            paddle.to_tensor([0.18994140625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_2cfe6882b81a9e69dcc5b24096966478(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1f9768dbc6ef556427844a92f14dca9
    def get_inputs(self):
        return [
            paddle.to_tensor([0.345703125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_f95854f8aefbf9a18d35a8ae07650072(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1f9768dbc6ef556427844a92f14dca9
    def get_inputs(self):
        return [
            paddle.to_tensor([0.359130859375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_13973f6224fa7b77be44a87abbce02fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1f9768dbc6ef556427844a92f14dca9
    def get_inputs(self):
        return [
            paddle.to_tensor([0.403076171875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_880e93af96ffe3567f0b4a28f9c30f59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1f9768dbc6ef556427844a92f14dca9
    def get_inputs(self):
        return [
            paddle.to_tensor([0.12548828125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_da2402627ede5a2afe2ee21b1b54d8b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1f9768dbc6ef556427844a92f14dca9
    def get_inputs(self):
        return [
            paddle.to_tensor([0.03143310546875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_dd97f944656ca4f2d9099f278786b0e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1f9768dbc6ef556427844a92f14dca9
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0244598388671875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_84ec94804e78911fefed64ee50ab4c0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1f9768dbc6ef556427844a92f14dca9
    def get_inputs(self):
        return [
            paddle.to_tensor([0.474365234375], dtype='float16').reshape([1]),
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

class PrimitiveOp_2c9db272ebe5c2fbfe613f847c689999(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 96, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f5c5fe7d41c1e50ede1aba38bb97564a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c9db272ebe5c2fbfe613f847c689999
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
class TestPrimitiveOp_13897c80d01e09917288bde42f65e23c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c9db272ebe5c2fbfe613f847c689999
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
class TestPrimitiveOp_6ea297e492fb24ba0d9996ab058f7752(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c9db272ebe5c2fbfe613f847c689999
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
class TestPrimitiveOp_68715e197d93f6ac92f00b01bd78a286(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c9db272ebe5c2fbfe613f847c689999
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

class PrimitiveOp_73745b9f9de4124db115b96ab02e063a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 24, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 24, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5729d27cedb1a4dfd7befad4257e38d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73745b9f9de4124db115b96ab02e063a
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
class TestPrimitiveOp_f8723ba43afd5f080f77723b31cfefd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73745b9f9de4124db115b96ab02e063a
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
class TestPrimitiveOp_9a999db5bf1e8c3162d84123900019e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73745b9f9de4124db115b96ab02e063a
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
class TestPrimitiveOp_6c888179bbe8b29cc4b5e6bf24259661(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73745b9f9de4124db115b96ab02e063a
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

class PrimitiveOp_ac0ae25d3f54746d7b9bb8b633be03e4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 72, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 72, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_517005102b8841a8392d7a6279bccdcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac0ae25d3f54746d7b9bb8b633be03e4
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

class PrimitiveOp_6bbfcb13635d0ee466e6ef38a6192283(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 120, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 120, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_474ab546fc3d77b4368bb0c4a8a1f369(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bbfcb13635d0ee466e6ef38a6192283
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
class TestPrimitiveOp_6d04a84365c7f78217e584aa90685bbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af8a58da019b9911d1523aab76aa63cd
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

class PrimitiveOp_f782d7dbea763a914af953ed3052afdd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 672, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 672, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_472558919bd5842492d78c05d0f245da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f782d7dbea763a914af953ed3052afdd
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
class TestPrimitiveOp_349f75f1451a3437f2d3c05736e43574(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f782d7dbea763a914af953ed3052afdd
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

class PrimitiveOp_934e944ba9c81a577462dce5e614d0d3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 960, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 960, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b66d67de311cddc9d754fab462458bbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_934e944ba9c81a577462dce5e614d0d3
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

class PrimitiveOp_b05329eda62ff081cbe6e3776b6426da(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, 1, 26], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 512, 1, 26], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_92db2aa24dce5bd5ce5728b49f4ccfc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b05329eda62ff081cbe6e3776b6426da
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

class PrimitiveOp_814a60e05aac82cfd6e5716056ee7275(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[320], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 196, 320], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b4743e26dc44699b62f9860d755620aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_814a60e05aac82cfd6e5716056ee7275
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

class PrimitiveOp_faf661519f5efdd0503134ff7584aeea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 49, 512], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_aa344de6a5f82a7d9b088332ae489edd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_faf661519f5efdd0503134ff7584aeea
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

class PrimitiveOp_ac274a4592dd04aa9d58450f9c246d0b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, 1, 26], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 512, 1, 26], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_390a1e507394a12127a6b973347a7b6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac274a4592dd04aa9d58450f9c246d0b
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

class PrimitiveOp_761c2d70f109f80d3370e3538fcfbe06(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 40, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 40, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_515d560d85d7904fe0c6464d56b4092f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_761c2d70f109f80d3370e3538fcfbe06
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

class PrimitiveOp_65d5686eae9778cbefbe578c5c4b1198(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 64, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9a8b3c71f11195d951981b71e6ab1d81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65d5686eae9778cbefbe578c5c4b1198
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

class PrimitiveOp_21bcfdf57b59139a84914ac95810b0e7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 240, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fdc6746d2e9bdb1ce16f0fc2b88f4e4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21bcfdf57b59139a84914ac95810b0e7
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

class PrimitiveOp_acfd809450c64d908f475fd4a49c1662(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 336, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 336, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8420ba8b86065552cb3c062e92bcc1db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_acfd809450c64d908f475fd4a49c1662
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
class TestPrimitiveOp_144f894b9753a9b81c63e38328b439ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_acfd809450c64d908f475fd4a49c1662
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
class TestPrimitiveOp_40c1d94452bb1e3d5979fadd2a2e9fb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af809f60aaa86de5f779f4a04db52410
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

class PrimitiveOp_300ee4ca9ce300430c11a0a8643969ba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7d883ed1123408e7f2b6e5d3e806aa6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_300ee4ca9ce300430c11a0a8643969ba
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 800, 1344], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[[[0.2438969910144806]], [[0.37806886434555054]], [[0.023386424407362938]]]], dtype='float32').reshape([1, 3, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_aae5c56eace724340840f8107f1c620d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_788941025d3968261a5b1aa84d229fd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aae5c56eace724340840f8107f1c620d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 100, 168], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4426266849040985], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fe009243919e354299431e02ef47e3c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aae5c56eace724340840f8107f1c620d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 50, 84], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.35133931040763855], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0183c55bf8f9b3ec4630055d2eb73a73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aae5c56eace724340840f8107f1c620d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 25, 42], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.21347853541374207], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d3ae4e46c5dad799d4660222af2c9ef6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aae5c56eace724340840f8107f1c620d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 13, 21], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.434526264667511], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4642118053ba0afa265865ba69aea77d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aae5c56eace724340840f8107f1c620d
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 7, 11], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4117897152900696], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_bcf389ab49cd0e55f3fc4a7425335e34(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e3b68a1aaa0352d5debd2b5128468ead(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcf389ab49cd0e55f3fc4a7425335e34
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
class TestPrimitiveOp_3716e79ff80afae959a39f80756dd5f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcf389ab49cd0e55f3fc4a7425335e34
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
class TestPrimitiveOp_38d6ce4c54837ecf31dca46188cfd13f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcf389ab49cd0e55f3fc4a7425335e34
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
class TestPrimitiveOp_c1f3ac7bb5f85d6fc7f9e03a287dd7ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcf389ab49cd0e55f3fc4a7425335e34
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
class TestPrimitiveOp_56172b708d7813a53cf6d2cb8af1f75a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcf389ab49cd0e55f3fc4a7425335e34
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

class PrimitiveOp_629dd5bf5b55f646448a9aa23e30d0cf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 32, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_197d623b6807e4cc2b73a066b663a74a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_629dd5bf5b55f646448a9aa23e30d0cf
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

class PrimitiveOp_d6a03c9e9f672fa64b851fbdde34c284(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 48, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7a7e198b9cb88e318e3cc418ccc8eee9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6a03c9e9f672fa64b851fbdde34c284
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

class PrimitiveOp_d3e6ff0af9bfd2369c5a99d45bf3798a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 64, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_119f36d991c09bdcf074c77c3f870de3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3e6ff0af9bfd2369c5a99d45bf3798a
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

class PrimitiveOp_7fd23225340276662c4acd31beccf327(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 128, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4a4b2441c9de89473316bde7787e455e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7fd23225340276662c4acd31beccf327
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

class PrimitiveOp_ca30f3d802133bf3c6491be560093a9a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 128, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f4dcb463279bd2957af21496e48c991c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca30f3d802133bf3c6491be560093a9a
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

class PrimitiveOp_b91790641d7b65854ac2413d7a311c6a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 256, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5918671c3e20a2e5c4a4ce0d1a0c021b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b91790641d7b65854ac2413d7a311c6a
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

class PrimitiveOp_2448fec144996eaef1b2fe0aa1e85556(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 196, 320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b4484cecab97000211f44f1008c0cad2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2448fec144996eaef1b2fe0aa1e85556
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

class PrimitiveOp_c2e9a678c31e9430d58e2c4853eef912(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 49, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_374075a3445a50d2c1f8acb7e12d45d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c2e9a678c31e9430d58e2c4853eef912
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

class PrimitiveOp_c003aa4970a4f0e1149bf1f171677986(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 26, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 26, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_70edf11915b5d9b00b42b7f4f602ec9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c003aa4970a4f0e1149bf1f171677986
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

class PrimitiveOp_e7eb74ac0e4551e9d8ff5b49a0cfbbb0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2, 1, 9, 112, 112], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 2, 16, 9, 112, 112], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5ddd62727e5f560c2d8635c8714072d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7eb74ac0e4551e9d8ff5b49a0cfbbb0
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

class PrimitiveOp_289a3bd13e1d7e2b6d645e92767499fd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 1, 49, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4, 16, 49, 56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_994a33d6d1b203f0f1778639ea38be67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_289a3bd13e1d7e2b6d645e92767499fd
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

class PrimitiveOp_9122447ecc7a31baee8307b1ee629ce6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, 1, 49, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 8, 16, 49, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cb57fbe245abcaa6365e04f917de9e49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9122447ecc7a31baee8307b1ee629ce6
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

class PrimitiveOp_679a42ec02dafb4ebc85f5b26d7f6dfe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 1, 49, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 16, 16, 49, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_658b0b6c9f7bdaf8413edf6ffcc98f1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_679a42ec02dafb4ebc85f5b26d7f6dfe
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

class PrimitiveOp_9e05b177a1814ec1ea1b742f7602a570(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 1, 49, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 32, 16, 49, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c49ff4d7ebe08a6b019dbaeba52dc701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e05b177a1814ec1ea1b742f7602a570
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

class PrimitiveOp_94cc6ee2d87c473c812db41a8fe68576(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 256, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cfe2640bfabdbac6b131e857176e5aa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94cc6ee2d87c473c812db41a8fe68576
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

class PrimitiveOp_c77bd5ef79c1d832650c694de7ed02ad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 512, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4b790e4c8eb4bb6c5eac5ed1674ce471(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c77bd5ef79c1d832650c694de7ed02ad
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
class TestPrimitiveOp_25798ed2fc35504ed980f35c0b0402cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d2693ec05df3906d17f8a6606a188e5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.054487742483615875], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_2da79c7eeaaa1d3427c2f12ec73b2cb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d2693ec05df3906d17f8a6606a188e5
    def get_inputs(self):
        return [
            paddle.to_tensor([0.21072040498256683], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_404a53b7fb396404ecb041e2bb889bf8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6ce0805f387f39d16e1535160edb408
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3370343744754791], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_32d41035412a54b708b02bd882f309b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6ce0805f387f39d16e1535160edb408
    def get_inputs(self):
        return [
            paddle.to_tensor([0.047739412635564804], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_e1ee6d85cfd8c2f24958a6bc094589a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6ce0805f387f39d16e1535160edb408
    def get_inputs(self):
        return [
            paddle.to_tensor([0.15943484008312225], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_346648bf3aa765f25d37a4e2a412ced1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb5dc535d535efb2f4d64d27da775830
    def get_inputs(self):
        return [
            paddle.to_tensor([0.45699337124824524], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_5b5d29877c7b1707445390e378cb2b52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb5dc535d535efb2f4d64d27da775830
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10705216974020004], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_663033ab919424d6c5b919f4a14b76b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb5dc535d535efb2f4d64d27da775830
    def get_inputs(self):
        return [
            paddle.to_tensor([0.15463751554489136], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_de91cb7ac2b8858ab82f3e8fd97fe3d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb5dc535d535efb2f4d64d27da775830
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3243096172809601], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_8d6d5a9c17dd454e260e8a5ecceef9a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb5dc535d535efb2f4d64d27da775830
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2100619077682495], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_e7b6f10ba8a85632d099e2bec0bca7a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb5dc535d535efb2f4d64d27da775830
    def get_inputs(self):
        return [
            paddle.to_tensor([0.30606868863105774], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_350d174ec6b3e316c0d0b56b506d420a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb5dc535d535efb2f4d64d27da775830
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4733254909515381], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_6e201cc4701b533a9631904ba0008440(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6af68e33730ac05b49bc920ca5d6efe
    def get_inputs(self):
        return [
            paddle.to_tensor([0.45036402344703674], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_738cfdca85fceec75221cd2a97246572(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6af68e33730ac05b49bc920ca5d6efe
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4187791049480438], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_e640100e7d769d3eb3f5868ddd191a1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6af68e33730ac05b49bc920ca5d6efe
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2859806716442108], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_51b10552de8ef0a901e80d06151d70e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6af68e33730ac05b49bc920ca5d6efe
    def get_inputs(self):
        return [
            paddle.to_tensor([0.49253273010253906], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_68c03c9105e5425c4d9772d634c36354(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6af68e33730ac05b49bc920ca5d6efe
    def get_inputs(self):
        return [
            paddle.to_tensor([0.29879528284072876], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_aed17242f53b064fbdfed72770b57dc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6af68e33730ac05b49bc920ca5d6efe
    def get_inputs(self):
        return [
            paddle.to_tensor([0.28024914860725403], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_b1c35821ce8e82e719f01fba69e2d5b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6af68e33730ac05b49bc920ca5d6efe
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2171127051115036], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_54a3e693fddf83a7e124950b48fec8cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5d420d811f3ab50968af0e00a0906eb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.205111563205719], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_b38e63fc1df97bbad23fea2a6de23fd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5d420d811f3ab50968af0e00a0906eb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3341596722602844], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_0b57f4ae8c9bcf875712c0e4d4c42507(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5d420d811f3ab50968af0e00a0906eb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.28429579734802246], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_f90f798393e673c84127fcbcb128f73c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5d420d811f3ab50968af0e00a0906eb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.016756678000092506], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_cd26c9759b0dbf2b18d69563be87d383(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5d420d811f3ab50968af0e00a0906eb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.26156091690063477], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_2d9beed581ee47847e48037faa4f0087(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5d420d811f3ab50968af0e00a0906eb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.14741328358650208], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_a6ee31c6feddc1d031ed1168788afeb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5d420d811f3ab50968af0e00a0906eb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.21399670839309692], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_25858dee51ae45771db3c56da80c923a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5d420d811f3ab50968af0e00a0906eb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.40264075994491577], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_7534bf157ffbfebd174ee1c225596125(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5d420d811f3ab50968af0e00a0906eb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.02259201742708683], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_3d42bda6cefccd5601e60d799f8782b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5d420d811f3ab50968af0e00a0906eb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.28167831897735596], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_8705974c326e4467e9b798aed27144e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5d420d811f3ab50968af0e00a0906eb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2459322214126587], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_55235714f02d4c5eccc8c949d88c6320(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5d420d811f3ab50968af0e00a0906eb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.42528459429740906], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_d724c30df9af6a206e4c73d2891850bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5d420d811f3ab50968af0e00a0906eb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2386317253112793], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_4d1d388304857c96fdc315ce08ef0cc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5d420d811f3ab50968af0e00a0906eb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1340242624282837], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_a73d37b06b701cc7c077f3ceff9fc23f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5d420d811f3ab50968af0e00a0906eb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.419644832611084], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_3b4f3f366b3b26d30a87aff719cbf354(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5d420d811f3ab50968af0e00a0906eb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2740958333015442], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_6c1f4280b8d749ba1ca3a5da1064cab7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5d420d811f3ab50968af0e00a0906eb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4016909897327423], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_ac1a5dd083fb0624e88d4aac85931816(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5d420d811f3ab50968af0e00a0906eb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2894291877746582], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_2008e072d312826a0e6b0feb58289ca1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e5d420d811f3ab50968af0e00a0906eb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2736552059650421], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_79ded6f53737680ce540c4ffe5ad4bf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d5112d9c946203be8f957691d36cce8
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
class TestPrimitiveOp_3b21ba2df6ddd0ebf96b87cb40819077(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3facc485d59a84ad63c81dd8401dcc51
    def get_inputs(self):
        return [
            paddle.to_tensor([0.228871151804924], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_e045a2404b7b3b435398a6ff819e28ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3facc485d59a84ad63c81dd8401dcc51
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1975477635860443], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_47320e174fee1e618aab9ced05d1dfd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3facc485d59a84ad63c81dd8401dcc51
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3624515235424042], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_ff58d2b5878df49ae6c6b39b36c2fe78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3facc485d59a84ad63c81dd8401dcc51
    def get_inputs(self):
        return [
            paddle.to_tensor([0.4675005376338959], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_fd339289402e7dfe5000be4bc5f91212(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_843c795c904b4dd89d0f9575184e65d2
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
class TestPrimitiveOp_fcfec00428ca2325eafa83a77c767f02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3facc485d59a84ad63c81dd8401dcc51
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1207498237490654], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_ccc55b9ed58aeb7a5b7305d7481a7cce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3facc485d59a84ad63c81dd8401dcc51
    def get_inputs(self):
        return [
            paddle.to_tensor([0.05389200150966644], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_628eb67b8d066338f31adfcbc07e4830(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3facc485d59a84ad63c81dd8401dcc51
    def get_inputs(self):
        return [
            paddle.to_tensor([0.28303852677345276], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_0c128f5562337a9eed468f8416bc1a97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3facc485d59a84ad63c81dd8401dcc51
    def get_inputs(self):
        return [
            paddle.to_tensor([0.23547503352165222], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_4d80fc75cfaf2ca6e06349496172e248(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3facc485d59a84ad63c81dd8401dcc51
    def get_inputs(self):
        return [
            paddle.to_tensor([0.44396933913230896], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_937588e702ba7c21793eead4be1815ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3facc485d59a84ad63c81dd8401dcc51
    def get_inputs(self):
        return [
            paddle.to_tensor([0.39385876059532166], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_ea4721a6d2102a30bfdf43ef85429d0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3facc485d59a84ad63c81dd8401dcc51
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2808399200439453], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_284cd3ad930eb9e3a845d75bddf5412e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3facc485d59a84ad63c81dd8401dcc51
    def get_inputs(self):
        return [
            paddle.to_tensor([0.011308269575238228], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_c9dde676e6e6d139949173a261d0db20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3facc485d59a84ad63c81dd8401dcc51
    def get_inputs(self):
        return [
            paddle.to_tensor([0.44586485624313354], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_ed3ce7dfc05ab5d2612605f799063857(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3facc485d59a84ad63c81dd8401dcc51
    def get_inputs(self):
        return [
            paddle.to_tensor([0.48435136675834656], dtype='float32').reshape([1]),
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
class TestPrimitiveOp_e03d310af881ce26f826dc47097e98a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_685215e39df696ab66134852db25f88e
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
class TestPrimitiveOp_f47cc94ff9c8192c9938295d59758ec3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_685215e39df696ab66134852db25f88e
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
class TestPrimitiveOp_d2aeac43f2ea4466d6ea853984f43a18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_685215e39df696ab66134852db25f88e
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
class TestPrimitiveOp_8a44c319a14b85dad5cfe48c6237e422(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_685215e39df696ab66134852db25f88e
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
class TestPrimitiveOp_0d238b1b215904594152093ad7884b43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2cf6a2f6cd41e801025143d99d97402
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
class TestPrimitiveOp_f78de86aac1ada8848d30951ece5bdc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2cf6a2f6cd41e801025143d99d97402
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
class TestPrimitiveOp_12aaa241b2ddd78d1b9d10f850aa2e69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2cf6a2f6cd41e801025143d99d97402
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
class TestPrimitiveOp_3a26c4bbd0ff10205fb072a5690a42ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2cf6a2f6cd41e801025143d99d97402
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
class TestPrimitiveOp_b80446c6ed86077471f825bd5e77cb4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_761c2d70f109f80d3370e3538fcfbe06
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
class TestPrimitiveOp_be7c6a25215bea6f443978a9ef1e636d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65d5686eae9778cbefbe578c5c4b1198
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
class TestPrimitiveOp_a7c7734cfdf83b2482bda2cafd4dcde9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21bcfdf57b59139a84914ac95810b0e7
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
class TestPrimitiveOp_0590620ceb05b272228e544111d69de2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_acfd809450c64d908f475fd4a49c1662
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
class TestPrimitiveOp_f5ea20d26bb97185b1c83ecd40c2cc7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_acfd809450c64d908f475fd4a49c1662
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
class TestPrimitiveOp_fe82538722e8d01c9cedeea87b92ab3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af809f60aaa86de5f779f4a04db52410
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
class TestPrimitiveOp_3f9ff16a76cc25ba9eb1778a5a5db8bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f67b0a595cb0c360323bb1876232fd34
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10919189453125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_6578631f11d8265bf022074396de8670(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f67b0a595cb0c360323bb1876232fd34
    def get_inputs(self):
        return [
            paddle.to_tensor([0.025909423828125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_33a01b81e0961779016ac31ae5a83196(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df4bd3f2745f3c372cf78d808f434ed0
    def get_inputs(self):
        return [
            paddle.to_tensor([0.299560546875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_571d9b0e3f0750f36e837944def046e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df4bd3f2745f3c372cf78d808f434ed0
    def get_inputs(self):
        return [
            paddle.to_tensor([0.317626953125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_23d6cb98c8043e0de82f36ba895acf87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df4bd3f2745f3c372cf78d808f434ed0
    def get_inputs(self):
        return [
            paddle.to_tensor([0.25146484375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_dae5e29c25af31bdc4ae93f1ec79d2f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76f3d06fd684035c4e26fc5c7d6ca147
    def get_inputs(self):
        return [
            paddle.to_tensor([0.25927734375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_80c741fb786f4a5b96b2f16ce0716797(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76f3d06fd684035c4e26fc5c7d6ca147
    def get_inputs(self):
        return [
            paddle.to_tensor([0.45263671875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_7998bf9ac117bce27796096ab5245abd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76f3d06fd684035c4e26fc5c7d6ca147
    def get_inputs(self):
        return [
            paddle.to_tensor([0.28955078125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_2db0df6229766bef74706f95294d4f39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76f3d06fd684035c4e26fc5c7d6ca147
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10296630859375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_20085158605488bdb4780803a0c14eb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76f3d06fd684035c4e26fc5c7d6ca147
    def get_inputs(self):
        return [
            paddle.to_tensor([0.293701171875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_66292a6609910c8af49c15aec3da5c2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76f3d06fd684035c4e26fc5c7d6ca147
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1744384765625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_12340f532d3504868367ceef68f13465(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76f3d06fd684035c4e26fc5c7d6ca147
    def get_inputs(self):
        return [
            paddle.to_tensor([0.08538818359375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_fb3e59c99f1acadc914af0d363af5216(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c0907bc0f8545b9b41bfefa0339dc28
    def get_inputs(self):
        return [
            paddle.to_tensor([0.40185546875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_aef37a8980fb4f83697712186f9d3da7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c0907bc0f8545b9b41bfefa0339dc28
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10150146484375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_27a499a47089a48466e4e961ab3b30dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c0907bc0f8545b9b41bfefa0339dc28
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3974609375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_8fcc6981ea0005df795654a351cb0bd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c0907bc0f8545b9b41bfefa0339dc28
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1925048828125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_041b69692482e73959ddc4120ac23e61(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c0907bc0f8545b9b41bfefa0339dc28
    def get_inputs(self):
        return [
            paddle.to_tensor([0.391845703125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_fb41897d42bedc1cdf5ef29110cdd167(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c0907bc0f8545b9b41bfefa0339dc28
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2890625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_ce0a7ad4dae3a284177f450135099820(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c0907bc0f8545b9b41bfefa0339dc28
    def get_inputs(self):
        return [
            paddle.to_tensor([0.466552734375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_1e481fef4d100f900cf9bb8e6da1b320(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24bf4640dbd46b3eeb546b06869acb92
    def get_inputs(self):
        return [
            paddle.to_tensor([0.10125732421875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_87e86a84d0c25639b72f2115beb538ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24bf4640dbd46b3eeb546b06869acb92
    def get_inputs(self):
        return [
            paddle.to_tensor([0.08502197265625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_0decccfe060f632306248607fb993c4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24bf4640dbd46b3eeb546b06869acb92
    def get_inputs(self):
        return [
            paddle.to_tensor([0.363037109375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_397d314d9e4195dbde2dcb9b1631634a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24bf4640dbd46b3eeb546b06869acb92
    def get_inputs(self):
        return [
            paddle.to_tensor([0.28173828125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_e6cde9a9420f4d12cec9396660e70b09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24bf4640dbd46b3eeb546b06869acb92
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1937255859375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_9b62986736bac2a3221286eb5d280b43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24bf4640dbd46b3eeb546b06869acb92
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1319580078125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_009c37c696874ef38a1ab3c2eff6f645(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24bf4640dbd46b3eeb546b06869acb92
    def get_inputs(self):
        return [
            paddle.to_tensor([0.281494140625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_dd16ca152a037e3e5aed337a9c796d99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24bf4640dbd46b3eeb546b06869acb92
    def get_inputs(self):
        return [
            paddle.to_tensor([0.476318359375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_2aaf549e5b92a2af625bf6d02efd7ff4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24bf4640dbd46b3eeb546b06869acb92
    def get_inputs(self):
        return [
            paddle.to_tensor([0.01165008544921875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_b8d5ab0d5fe5d8dd2df2169cdd3be175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24bf4640dbd46b3eeb546b06869acb92
    def get_inputs(self):
        return [
            paddle.to_tensor([0.212158203125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_42c2a560911fa399b57150ed16dbdda9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24bf4640dbd46b3eeb546b06869acb92
    def get_inputs(self):
        return [
            paddle.to_tensor([0.380615234375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_2d9ceb763c58947df2eada4693c13a1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24bf4640dbd46b3eeb546b06869acb92
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1824951171875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_5494ba0e248206e53d75d4b55e76be4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24bf4640dbd46b3eeb546b06869acb92
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2254638671875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_9a23eb4e76afe056a59274fd3fab3dcc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24bf4640dbd46b3eeb546b06869acb92
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0010156631469726562], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_d950500598c40cb8fd89343cc5e14134(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24bf4640dbd46b3eeb546b06869acb92
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1300048828125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_d82692ba9fb5abdfe7b34f30493d08ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24bf4640dbd46b3eeb546b06869acb92
    def get_inputs(self):
        return [
            paddle.to_tensor([0.196044921875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_389890b9319580f0d4cc28b340e297ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24bf4640dbd46b3eeb546b06869acb92
    def get_inputs(self):
        return [
            paddle.to_tensor([0.3515625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_baae0225fd34bf6179bcdd0dfe9e070e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24bf4640dbd46b3eeb546b06869acb92
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2130126953125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_ce0ac06ffe8fb22849e0a5a1a7a2e4bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24bf4640dbd46b3eeb546b06869acb92
    def get_inputs(self):
        return [
            paddle.to_tensor([0.26806640625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_778fd2205f1b32b037c90e6d2e0741b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90d90adf304dfd9c18a5d81d20de13bd
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
class TestPrimitiveOp_42955e07bffc52b8a3f733ffc735bc13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1f9768dbc6ef556427844a92f14dca9
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1673583984375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_b552dd28956c6b99fe62070c5e2cbf68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1f9768dbc6ef556427844a92f14dca9
    def get_inputs(self):
        return [
            paddle.to_tensor([0.1259765625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_43c7f67e5b7e1f7c696a7b2a54144fde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1f9768dbc6ef556427844a92f14dca9
    def get_inputs(self):
        return [
            paddle.to_tensor([0.421875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_f0035bc8ce01c060bbd46f13bca023b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1f9768dbc6ef556427844a92f14dca9
    def get_inputs(self):
        return [
            paddle.to_tensor([0.487060546875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_ee7165776568be2ccbf376b569eb0f2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4262ebfd178c59cfbc57a0194b518a95
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
class TestPrimitiveOp_50c0a71d5078f32c9ccf90086695b33e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1f9768dbc6ef556427844a92f14dca9
    def get_inputs(self):
        return [
            paddle.to_tensor([0.241943359375], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_19decb62a2517f3110375a98dc24214d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1f9768dbc6ef556427844a92f14dca9
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0538330078125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_0eb6381cb9fec1c2d794ca60436b06e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1f9768dbc6ef556427844a92f14dca9
    def get_inputs(self):
        return [
            paddle.to_tensor([0.219482421875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_5a4205dda737997f8ec1ce5560aad2e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1f9768dbc6ef556427844a92f14dca9
    def get_inputs(self):
        return [
            paddle.to_tensor([0.369873046875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_ad317c069edda719f4295194ccb4943f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1f9768dbc6ef556427844a92f14dca9
    def get_inputs(self):
        return [
            paddle.to_tensor([0.499267578125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_31fd5d20080743334f0b1723e78d2c97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1f9768dbc6ef556427844a92f14dca9
    def get_inputs(self):
        return [
            paddle.to_tensor([0.314453125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_d3314177003c005bc8fc32832d4bdb70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1f9768dbc6ef556427844a92f14dca9
    def get_inputs(self):
        return [
            paddle.to_tensor([0.36279296875], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_78004d0306f5b7db301570af58f7ff74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1f9768dbc6ef556427844a92f14dca9
    def get_inputs(self):
        return [
            paddle.to_tensor([0.161376953125], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_b869af5760e563baa02048bb25c0229d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1f9768dbc6ef556427844a92f14dca9
    def get_inputs(self):
        return [
            paddle.to_tensor([0.09332275390625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_1831d89805afa7b0e987d5534d1d6e4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1f9768dbc6ef556427844a92f14dca9
    def get_inputs(self):
        return [
            paddle.to_tensor([0.070556640625], dtype='float16').reshape([1]),
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
class TestPrimitiveOp_4b69e91df987dcbf67a1ed65aeb6b068(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c9db272ebe5c2fbfe613f847c689999
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
class TestPrimitiveOp_df2d6753e6ec1e0daacb1a8e8abb098b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c9db272ebe5c2fbfe613f847c689999
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
class TestPrimitiveOp_3788721011c4bb1b6ad0eb899874a1f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c9db272ebe5c2fbfe613f847c689999
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
class TestPrimitiveOp_0488670d77fff2cfb77e7a8df341fc1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c9db272ebe5c2fbfe613f847c689999
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
class TestPrimitiveOp_2fcaaf49a8736a4a257a41fda4cef90a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73745b9f9de4124db115b96ab02e063a
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 20, 20], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([[[[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]]]], dtype='float16').reshape([1, 24, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_687f81d27fc2300b7760cc8fe9ac1129(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73745b9f9de4124db115b96ab02e063a
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
class TestPrimitiveOp_5c2e2c865ae17f6bbb0b2374cfdd9f39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73745b9f9de4124db115b96ab02e063a
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
class TestPrimitiveOp_8618598fb8e4888bd0c0ef93477eecd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73745b9f9de4124db115b96ab02e063a
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

class PrimitiveOp_3d5a52b839a3b22bd72504726205d901(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[32, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 32, 56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cce3b6230ae447f0cf3695b56465c790(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d5a52b839a3b22bd72504726205d901
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

class PrimitiveOp_ad087e83f867fafc980165221df9c12e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[64, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 64, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5627bc27b69faac329f987552ffb24b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad087e83f867fafc980165221df9c12e
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

class PrimitiveOp_c7104f409de5b5b805d8dfcf9e649691(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[160, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 160, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_89c622332c975093f038c653afb56530(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7104f409de5b5b805d8dfcf9e649691
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

class PrimitiveOp_74886718b79a9d9753e19f3ac1dc70d4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[256, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 256, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_188829789b224d885538decf76770064(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74886718b79a9d9753e19f3ac1dc70d4
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
class TestPrimitiveOp_a9a5d97e1df2e2aeacfabc77d3e307bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c13876966c9e62bb2b8d3893967019ec
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
class TestPrimitiveOp_fc24f17407667acb679295084a01f924(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f47f8cde93bde2463ad8c9966f0858bf
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
class TestPrimitiveOp_fc9bf870fca3b5b4142bd31d7e387485(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_07bf753e32b01b76520d7697121f07ad
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
class TestPrimitiveOp_096918ce89b0dc0df2238fe9a40c8b97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb7533bfe2ab31d32437f21c72cd99f3
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
class TestPrimitiveOp_118a51c1e29c05376fa4aebde64733f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb7533bfe2ab31d32437f21c72cd99f3
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
class TestPrimitiveOp_ec977f4cd7fe339a99bc70adfb5243fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af8a58da019b9911d1523aab76aa63cd
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

class PrimitiveOp_3a37eed7c7564dc868b4e497676dca73(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 384, 20, 20], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 384, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_86e22a8447fbeaa1f9d62aeb9021cfb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a37eed7c7564dc868b4e497676dca73
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

class PrimitiveOp_ad5e55c34817055f9723c93ee6c71e93(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, 40, 40], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 192, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_47e88d1743193a396f98c28c52b5c13f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad5e55c34817055f9723c93ee6c71e93
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

class PrimitiveOp_2b5b4b2f5f48de2e7a2af5e75f1e4526(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return input_0 * input_1

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 80, 80], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 96, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d2e6fa6a17a00abecfc488c599b58f85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2b5b4b2f5f48de2e7a2af5e75f1e4526
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