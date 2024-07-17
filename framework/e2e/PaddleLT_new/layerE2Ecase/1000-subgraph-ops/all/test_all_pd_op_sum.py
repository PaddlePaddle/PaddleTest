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
class PrimitiveOp_853876122220eea0d63c9159daec82fa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [3]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0c6a50fa25a59cac88eb22a14287833c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_853876122220eea0d63c9159daec82fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 16, 9, 112, 112], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dd0d4488b0cdbe0075f366ac87af3056(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([4392], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ebad67407114cd2888246deb2b826dc7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7dd26db87d3126260ac6c5284eafed0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ebad67407114cd2888246deb2b826dc7
    def get_inputs(self):
        return [
            paddle.uniform([1, 8732, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6b7cc780f7375c545c055b32201efba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
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

class PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_33927de077cdab2d962252d026ffb3c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4b93cdba544cb6692b7bcace7400b69b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9e725f8f16dcc98865b16222e2aef71e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b93cdba544cb6692b7bcace7400b69b
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f085f9c572f40c97d11eed0bd1e00508(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b93cdba544cb6692b7bcace7400b69b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.164689302444458, 0.017757881432771683]], [[0.043633297085762024, 0.10437674075365067]], [[0.0073349978774785995, 0.011525948531925678]], [[0.0007443867507390678, 1.684521407696593e-06]], [[0.029950877651572227, 0.00021591379481833428]], [[0.1846202313899994, 0.006817847024649382]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2cb355e7a729d679fb53f74f6b926be6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b93cdba544cb6692b7bcace7400b69b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.061260852962732315, 0.01057954877614975]], [[0.00023561378475278616, 0.08047422021627426]], [[0.10999879240989685, 0.018358053639531136]], [[0.002051433315500617, 0.006653859280049801]], [[0.038056258112192154, 0.06842878460884094]], [[0.1137068048119545, 7.871824345784262e-05]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_41c073ede18dc36ce782f3dfd37d123d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_853876122220eea0d63c9159daec82fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 16, 49, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c285596cb12d67cb2c1e3d77899a3e70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.to_tensor([0.23476628959178925, 0.07280173152685165, 0.1527838408946991, 0.06264255195856094, 0.1365392804145813, 0.029590526595711708, 0.05223521962761879, 0.19382347166538239, 0.1395426094532013, 0.08002318441867828, 0.21484430134296417, 0.06148171052336693, 0.22627988457679749, 0.0059169502928853035, 0.1382962316274643, 0.23641149699687958], dtype='float32').reshape([16]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8502292ffd745351eba85a39c28e06a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8691ccc3f3cbe682d32db5370c1fd112(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([150], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_acff5bba89495967566948b9e27d00e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b14cb593195ebde69945d4c84691815a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bc58357d1c8bc9d89a25daa555028727(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b14cb593195ebde69945d4c84691815a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3549, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_725c113bba128dfd2087f1880d360fc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([1790, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eff2da3f63346f42401b6d4315c7f25b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([15200], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_192a11c88ad5bbd7dafebbd04cdc4fe6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0793503075838089, 0.04147496819496155, 0.4323243796825409, 0.046486854553222656], [0.19966381788253784, 0.08242802321910858, 0.22307482361793518, 0.29298198223114014], [0.1793961077928543, 0.04817909002304077, 0.1912694275379181, 0.1161852478981018], [0.29603123664855957, 0.0002084672451019287, 0.2556636333465576, 0.09453093260526657], [0.17138390243053436, 0.3351079523563385, 0.07035799324512482, 0.04384499788284302]], dtype='float32').reshape([5, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6c24d4a1aeec5dd83fd1965f4125950d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_853876122220eea0d63c9159daec82fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 4, 16, 49, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c233aa15872e28f02473df12d8613bc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.24803951382637024, 0.20552822947502136, 0.026768580079078674, 0.3316788673400879], [0.18803435564041138, 0.20201504230499268, 0.20017105340957642, 0.20067209005355835], [0.4473848342895508, 0.05407550930976868, 0.017409682273864746, 0.05301254987716675], [0.18803435564041138, 0.20201504230499268, 0.20017105340957642, 0.20067209005355835], [0.4473848342895508, 0.05407550930976868, 0.017409682273864746, 0.05301254987716675]], dtype='float32').reshape([5, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1bd865dfe9ef91ef80c8c0838d5db4c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ebad67407114cd2888246deb2b826dc7
    def get_inputs(self):
        return [
            paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2d8e23a214aab71809c8444277716859(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b14cb593195ebde69945d4c84691815a
    def get_inputs(self):
        return [
            paddle.uniform([1, 11109, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_76f6133cb77cd3e2fe1d556260382e13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([5590, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_38f10feb3c4f1e57b93dadbb7e9631fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.02271851897239685, 0.34917765855789185, 0.28184565901756287, 0.31205257773399353], [0.32825154066085815, 0.09454698860645294, 0.31206318736076355, 0.13202887773513794], [0.1960485726594925, 0.26003414392471313, 0.26771461963653564, 0.3788975179195404], [0.32825154066085815, 0.09454698860645294, 0.31206318736076355, 0.13202887773513794], [0.1960485726594925, 0.26003414392471313, 0.26771461963653564, 0.3788975179195404], [0.241317480802536, 0.17325927317142487, 0.21977493166923523, 0.144002765417099], [0.241317480802536, 0.17325927317142487, 0.21977493166923523, 0.144002765417099]], dtype='float32').reshape([7, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_54936e165dd69b02828cad09c77012eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4151e62d366d2d9f35f80f1a184f4597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_853876122220eea0d63c9159daec82fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4f88d7b6c1b03048c675c3dec62346c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_554fe47f90c849d9de2c9dd1f5604882(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_853876122220eea0d63c9159daec82fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f7fd45bc280da1ba5ffd736f120d3bd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.to_tensor([0.00620089890435338, 0.12085138261318207, 0.15618959069252014, 0.20312778651714325, 0.1213756576180458, 0.195688858628273, 0.19383911788463593, 0.2417881041765213, 0.19907361268997192, 0.2289368361234665, 0.18400008976459503, 0.18927527964115143, 0.01187081541866064, 0.26018375158309937, 0.1211872547864914, 0.20321908593177795, 0.021043997257947922, 0.07587876170873642, 0.027009081095457077, 0.15242215991020203, 0.2501938045024872, 0.2517254948616028, 0.14631694555282593, 0.21730951964855194], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_28b34ad5d1846cebf0f7cd1c018b8c7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b14cb593195ebde69945d4c84691815a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3024, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1ab8d3c39686ea150bf2baca3eec9704(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([1532, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_34b1a3bc6b75cffe7654c4e19196680c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_853876122220eea0d63c9159daec82fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1590824e3c6f4300695677d90ba8f74b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.to_tensor([0.04771742597222328, 0.1647259145975113, 0.1667053997516632, 0.032060571014881134], dtype='float32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3805891f0348ddcff9d08e4a94d13b17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4052853584289551, 0.2120211124420166, 0.13143542408943176, 0.11902055889368057], [0.3741813898086548, 0.15131334960460663, 0.25213176012039185, 0.2869222164154053], [0.45656049251556396, 0.03858155757188797, 0.2733534574508667, 0.05612369626760483], [0.1426851451396942, 0.12330485880374908, 0.23283857107162476, 0.10144320130348206], [0.1426851451396942, 0.12330485880374908, 0.23283857107162476, 0.10144320130348206], [0.45656049251556396, 0.03858155757188797, 0.2733534574508667, 0.05612369626760483]], dtype='float32').reshape([6, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8eeba4905d66e24a322a43cdb028521c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.20082534849643707, 0.32808083295822144, 0.07644303143024445, 0.30330777168273926], [0.0014493167400360107, 0.028126269578933716, 0.23886965215206146, 0.06523522734642029], [0.1258922517299652, 0.17983423173427582, 0.2320423424243927, 0.09538260102272034], [0.18720334768295288, 0.03901803493499756, 0.2975650131702423, 0.0439382903277874], [0.20082534849643707, 0.32808083295822144, 0.07644303143024445, 0.30330777168273926]], dtype='float32').reshape([5, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cc46a9926d338223304783a1e281272e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6f057bee27710213eece533c08e54c23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.18431055545806885, 0.2801476716995239, 0.2272026538848877, 0.4437217116355896], [0.3008652329444885, 0.30532917380332947, 0.19709378480911255, 0.3049372434616089], [0.003745894879102707, 0.33901554346084595, 0.2402176707983017, 0.14623937010765076], [0.12322506308555603, 0.022577501833438873, 0.4758502244949341, 0.3571644425392151]], dtype='float32').reshape([4, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_380643f43a83add64561a2f1f1295527(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2, 3]
        return paddle._C_ops.sum(input_0, input_1, None, True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c906e6ddb17e1e38fc07834efd0d9b0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_380643f43a83add64561a2f1f1295527
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 34], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ef012bb61ab49ae618b23b581beb16b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9d2d313922bcf6debd7c96ab53cbe884(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([950], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_34964b306f6aff3eef08681a2fb88f88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([8816], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fc8f1547af4dc44b51b16e846b0d7765(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b14cb593195ebde69945d4c84691815a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e974666fa64735950a30a1503e2df1c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([2029, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_11c0443456334565a37ed7f954128799(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_380643f43a83add64561a2f1f1295527
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 152, 272], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_035d8f45057d1e59270691ea6a831a79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.12446340918540955, 0.36414214968681335, 0.005864471197128296, 0.29803407192230225], [0.12446340918540955, 0.36414214968681335, 0.005864471197128296, 0.29803407192230225], [0.10262435674667358, 0.1355656087398529, 0.005740270018577576, 0.1576295793056488], [0.07281385362148285, 0.05830588936805725, 0.07737109065055847, 0.06039551645517349], [0.02050727605819702, 0.17510975897312164, 0.46132177114486694, 0.1165611743927002], [0.03253069519996643, 0.2666469216346741, 0.1786847710609436, 0.2645733654499054], [0.15054616332054138, 0.12784788012504578, 0.14889222383499146, 0.13332661986351013]], dtype='float32').reshape([7, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8d8570e694ab13a7dd129024770c7527(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b14cb593195ebde69945d4c84691815a
    def get_inputs(self):
        return [
            paddle.uniform([1, 9261, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3fe1413fbb0e79ac7bf3228ffc3cf1d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([4671, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_49d0c1e920c37fe80a65eea39f4a1bab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([4842], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1e46e28287a73cb6bc5543240a220dea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([1192], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_36f9988c825be420e92a28ce54fb0c39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ebad67407114cd2888246deb2b826dc7
    def get_inputs(self):
        return [
            paddle.uniform([1, 2434, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_33be465f5a1a52ea9e591e283e395409(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b14cb593195ebde69945d4c84691815a
    def get_inputs(self):
        return [
            paddle.uniform([1, 2100, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_06d964f9c27970e541453696710eb11f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([1040, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bf8e3dcc6382baef2da2bde3703883ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.05943205952644348, 0.11794731020927429, 0.03552805632352829, 0.32374459505081177], [0.009605452418327332, 0.3527609705924988, 0.2734294533729553, 0.23799994587898254], [0.009605452418327332, 0.3527609705924988, 0.2734294533729553, 0.23799994587898254], [0.24064888060092926, 0.011556893587112427, 0.09050866961479187, 0.34660637378692627], [0.03751459717750549, 0.06767889857292175, 0.029917988926172256, 0.3286154270172119], [0.25522735714912415, 0.43620944023132324, 0.040089961141347885, 0.17711180448532104]], dtype='float32').reshape([6, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b82c20508a3746fdfe2b256b33935690(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b14cb593195ebde69945d4c84691815a
    def get_inputs(self):
        return [
            paddle.uniform([100, 2, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6e3ba821d800f6f00c78e8d01727cf98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_853876122220eea0d63c9159daec82fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f882a4a38e4241625cedaf8ace8f1902(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b14cb593195ebde69945d4c84691815a
    def get_inputs(self):
        return [
            paddle.uniform([300, 2, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_253f2243c39191e9468c3e7bb0943cca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b14cb593195ebde69945d4c84691815a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4725, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1a6b8d440b10c936ab8d04abc78b2adf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_544ff788c6cf5b7ebf62ec54ee9d0c32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b14cb593195ebde69945d4c84691815a
    def get_inputs(self):
        return [
            paddle.uniform([1, 6069, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_496bbb20ca3af5ef07eb789d49743868(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([3043, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a0554bbbb9190a7fdeb438f1e3082b1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b14cb593195ebde69945d4c84691815a
    def get_inputs(self):
        return [
            paddle.uniform([1, 7581, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5f54a72cb01949481ae1cd110835f4f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([3752, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_157735f0fc8298cb82be053c539afb36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([247], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_70db61c3ddd2476f482f20c931083a79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_853876122220eea0d63c9159daec82fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_69418e20dda696d6b8e63fc16e67e059(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_853876122220eea0d63c9159daec82fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3d1f12b594c907c4aced22100469e9f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.to_tensor([0.048170290887355804, 0.09018824994564056, 0.23842187225818634, 0.08789439499378204, 0.17725223302841187, 0.24263249337673187, 0.01934143528342247, 0.05695542320609093, 0.022073082625865936, 0.08618254959583282, 0.17055098712444305, 0.029788516461849213, 0.1458057314157486, 0.2550959289073944, 0.13721953332424164, 0.25281277298927307, 0.06536693871021271, 0.18614676594734192, 0.15816795825958252, 0.1805841028690338], dtype='float32').reshape([20]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cf44653fc9d63d624fc547e42a2d092a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([17508], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_067fe73dc7ed04ffa49a44e06df2f2f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([70], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_820d9d57504738da6fe66500d5533b70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f763aad3da82bd205a307cff7d2a7469(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b14cb593195ebde69945d4c84691815a
    def get_inputs(self):
        return [
            paddle.uniform([1, 4116, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_edcdad011bf1221672ede52fc62fd0f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([2058, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_267a49e9e4ec3ddab8b36e1230e1a369(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_853876122220eea0d63c9159daec82fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 2, 16, 9, 112, 112], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_966a7dde525286f5a72039c06600fbcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([551], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3ab8e0ceff03873c56b74c57caa63708(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.017386049032211304, 0.33063408732414246, 0.1257045716047287, 0.04128274321556091], [0.3253510594367981, 0.024507902562618256, 9.924173355102539e-06, 0.08920297026634216], [0.16159535944461823, 0.09419108927249908, 0.16095225512981415, 0.2541719675064087], [0.16159535944461823, 0.09419108927249908, 0.16095225512981415, 0.2541719675064087], [0.31531959772109985, 0.20655792951583862, 0.08761733770370483, 0.1272023618221283]], dtype='float32').reshape([5, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b64a20a0d0b6b81fb55369237a0bd9ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([3800], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ac42641436898515102dc216a48bbc54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([2204], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f2900eb4c54d8be40362ea07d7499409(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8760fc00fdd043699fe994dd526ce471(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b14cb593195ebde69945d4c84691815a
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_17916b908b31932d58db4f0261752b2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([4175, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c7b213901e66e62738bbb84d5bae8641(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.028999531641602516, 0.018607884645462036, 0.14095304906368256, 0.367298424243927], [0.002734735608100891, 0.10281184315681458, 0.07219496369361877, 0.2548578679561615], [0.007467515766620636, 0.21890367567539215, 0.03665885329246521, 0.17195095121860504], [0.028999531641602516, 0.018607884645462036, 0.14095304906368256, 0.367298424243927], [0.01756387948989868, 0.001589447259902954, 0.27317219972610474, 0.3700735867023468], [0.2575816512107849, 0.14444348216056824, 0.2816029191017151, 0.01001708209514618], [0.01756387948989868, 0.001589447259902954, 0.27317219972610474, 0.3700735867023468]], dtype='float32').reshape([7, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_95ebdfded8b4e17099d5b78f4e2856e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([52, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f581d793b37c655276bec41e44874947(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [3]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 2, 16, 9, 112, 112], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4a23d9dd91b6eca60f73053bda05c5fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f581d793b37c655276bec41e44874947
    def get_inputs(self):
        return [
            paddle.uniform([10, 2, 16, 9, 112, 112], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a1943fcd467c716c9c9fe98068bbabb6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [3]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 4, 16, 49, 56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4219ebd0a0896c08ebf3130451e09752(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1943fcd467c716c9c9fe98068bbabb6
    def get_inputs(self):
        return [
            paddle.uniform([10, 4, 16, 49, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_27c306d248877430ed152c9699bd02ec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = []
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_856d399859d49668c1c7e88b657a3de1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([1790, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_947fa2b3364eaa2221440ee79e1fe59b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [3]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 4, 16, 49, 56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b22eef7c9f407dd3ee42df1ed337d6cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_947fa2b3364eaa2221440ee79e1fe59b
    def get_inputs(self):
        return [
            paddle.uniform([22, 4, 16, 49, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2751ba41a94f13207ce7b2bf9234d1f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([5590, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_353023a9e1777c9fb64f3d4114b0d44e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [3]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 32, 16, 49, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c9febe40f6b3a88feb19e03842bb1071(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_353023a9e1777c9fb64f3d4114b0d44e
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_61e10fd16de6fe0c6eb3adaf4edccc33(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [3]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 8, 16, 49, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5fc384a0313ff2c77e7833e2e0456f10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61e10fd16de6fe0c6eb3adaf4edccc33
    def get_inputs(self):
        return [
            paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ae89a696712cbc23fb762db86bb30bd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([1532, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_bc93a4dd9fbefeb7bc3815e44d0d4e36(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [3]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 16, 16, 49, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5e478cbf7e713aa4afaef2f7421c9986(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bc93a4dd9fbefeb7bc3815e44d0d4e36
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5b341617cee51e0d227eb01c08b01589(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([2029, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7321f3e098e8b81162f3a676a7470d7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([4671, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9586bb909ae71abe04dc83669c6d56d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([1040, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_347351c4052d55ea212d962433a77e40(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a46897f450a7b133e92d7fd0864ad502(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_347351c4052d55ea212d962433a77e40
    def get_inputs(self):
        return [
            paddle.uniform([100, 2, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a7e08bca6ef5b9395a1c66b765eab6ba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [3]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 32, 16, 49, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_55055cf689aa537db9ac3989b4e51f0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7e08bca6ef5b9395a1c66b765eab6ba
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_10b6046c666418323127d683f805ec6e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7dbe55b5f22236ec45e2c125e506baaa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10b6046c666418323127d683f805ec6e
    def get_inputs(self):
        return [
            paddle.uniform([300, 2, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_66ed2e0752bbf2164146f70e3a72c7a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0a8add0840320fb99fa217523ba3eccb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([3043, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5193e0c277cc4d4006ac239993865584(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([3752, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ecd6d3da0633117aa68085eb55e58e40(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [3]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10, 8, 16, 49, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fe41bc4150a89fbeedc50e71dc034097(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ecd6d3da0633117aa68085eb55e58e40
    def get_inputs(self):
        return [
            paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_71fcd388c0490d458a059efab7644a00(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [3]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 16, 16, 49, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_45c827a7ac8be3216c30c8081aeb3632(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71fcd388c0490d458a059efab7644a00
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1be781fc9faf604f8bda4b8eb6e40456(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([2058, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e177028f6bbdd633baf19a6ccc427651(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [3]
        return paddle._C_ops.sum(input_0, input_1, None, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[22, 2, 16, 9, 112, 112], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8146c68152bd110a924d07031de2b6cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e177028f6bbdd633baf19a6ccc427651
    def get_inputs(self):
        return [
            paddle.uniform([22, 2, 16, 9, 112, 112], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6e9ccf7847d523009a4604957829163c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([4175, 1], dtype='float32', min=0, max=0.5),
        ]


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