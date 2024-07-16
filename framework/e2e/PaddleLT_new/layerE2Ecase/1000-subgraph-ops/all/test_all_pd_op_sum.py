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
        return True, f"last stage failed. stderr: {stderr}"
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
counter = itertools.count()
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
class TestPrimitiveOp_40c756c4b9ae16e0be708adfeca87990(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([4364], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eddc9067d5c58f77ea42db9e7698a1bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b93cdba544cb6692b7bcace7400b69b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.04490616172552109, 0.0004491924191825092]], [[0.017281103879213333, 0.031092826277017593]], [[0.0033467772882431746, 0.005517947021871805]], [[0.00017468284931965172, 0.024215685203671455]], [[0.025338083505630493, 0.20364879071712494]], [[0.001200223807245493, 0.05960644781589508]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8760642330c33b2cd88c70b973c90a6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b93cdba544cb6692b7bcace7400b69b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.008221765980124474, 0.10027185082435608]], [[0.005742238834500313, 0.05414322391152382]], [[0.015191194601356983, 0.0813697800040245]], [[0.0704416036605835, 0.0590856559574604]], [[0.0014381229411810637, 0.07639468461275101]], [[9.467014024266973e-06, 0.001696201041340828]]]], dtype='float32').reshape([1, 6, 1, 2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_27ad8c2f144669118ca0d1b89a4712d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.to_tensor([0.18588557839393616, 0.09749448299407959, 0.20484738051891327, 0.2216888666152954, 0.022168874740600586, 0.2648043930530548, 0.10762902349233627, 0.027730770409107208, 0.024402687326073647, 0.08377330750226974, 0.21219663321971893, 0.0921526849269867, 0.1076374277472496, 0.15573976933956146, 0.22562327980995178, 0.08805708587169647], dtype='float32').reshape([16]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_209ebdaa55a59f374818d23dc4a1d7a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([1812, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fdca632f257e5618ee3ddbcdbc4ae5e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.030847802758216858, 0.1512724906206131, 0.39145582914352417, 0.0012667477130889893], [0.27543580532073975, 0.07220971584320068, 0.3511688709259033, 0.18944287300109863], [0.3136575520038605, 0.07451048493385315, 0.2333432137966156, 0.34193605184555054], [0.3953903913497925, 0.09340949356555939, 0.13210399448871613, 0.20949581265449524], [0.13511842489242554, 0.20150819420814514, 0.3850752115249634, 0.11844278872013092]], dtype='float32').reshape([5, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c9f2b75c3101a77d6337ed31084eb6ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.23857006430625916, 0.2494465708732605, 0.26256534457206726, 0.11029650270938873], [0.15996772050857544, 0.02149178832769394, 0.1710740625858307, 0.1583121120929718], [0.03202125057578087, 0.009222090244293213, 0.2052132934331894, 0.2637884318828583], [0.15996772050857544, 0.02149178832769394, 0.1710740625858307, 0.1583121120929718], [0.03202125057578087, 0.009222090244293213, 0.2052132934331894, 0.2637884318828583]], dtype='float32').reshape([5, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8e0c721434614ff2584e14bc6d1aa3b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([5519, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a16e392b5938d3f3bc1b5471f7647205(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3680111765861511, 0.01847441866993904, 0.05460155010223389, 0.015949957072734833], [0.26094627380371094, 0.1204889714717865, 0.2633415460586548, 0.11405646800994873], [0.1106850802898407, 0.43407687544822693, 0.04854273796081543, 0.12346945703029633], [0.26094627380371094, 0.1204889714717865, 0.2633415460586548, 0.11405646800994873], [0.1106850802898407, 0.43407687544822693, 0.04854273796081543, 0.12346945703029633], [0.07249720394611359, 0.2051759958267212, 0.12682610750198364, 0.1690688580274582], [0.07249720394611359, 0.2051759958267212, 0.12682610750198364, 0.1690688580274582]], dtype='float32').reshape([7, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_661f6c1c2c335b0f81cae717c15927ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([1798, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3c76d52d47a592431efce40bb4376e01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.to_tensor([0.13009996712207794, 0.009651090949773788, 0.006956066004931927, 0.2358129918575287, 0.2666492462158203, 0.10272689163684845, 0.16449642181396484, 0.08720695972442627, 0.06186715513467789, 0.23609867691993713, 0.1194843128323555, 0.11775617301464081, 0.14119727909564972, 0.16470958292484283, 0.21151041984558105, 0.05700733885169029, 0.02420252002775669, 0.057465195655822754, 0.17149952054023743, 0.022391049191355705, 0.13238608837127686, 0.07877877354621887, 0.14951328933238983, 0.015134080313146114], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_775ba0829672ebcec0026684c5778832(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([1574, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1118301c3a9bc8dbc85b534302a2fc89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.to_tensor([0.08558669686317444, 0.1315050721168518, 0.185883030295372, 0.025148898363113403], dtype='float32').reshape([4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_433b2e0e6a51f0b9fc71d5e9bfdc3028(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.14555048942565918, 0.005882769823074341, 0.11834472417831421, 0.1643039435148239], [0.2269618809223175, 0.2803482413291931, 0.0850462019443512, 0.1952138990163803], [0.024031832814216614, 0.08829766511917114, 0.10472142696380615, 0.012247443199157715], [0.23522406816482544, 0.149470716714859, 0.23910215497016907, 0.02997654676437378], [0.23522406816482544, 0.149470716714859, 0.23910215497016907, 0.02997654676437378], [0.024031832814216614, 0.08829766511917114, 0.10472142696380615, 0.012247443199157715]], dtype='float32').reshape([6, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a3ea78f07a5f37fdd31b42a2d142fbd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.20295970141887665, 0.03882686793804169, 0.0032911300659179688, 0.18110018968582153], [0.10731416940689087, 0.11742308735847473, 0.39797326922416687, 0.13853870332241058], [0.0022506117820739746, 0.3054242432117462, 0.15239760279655457, 0.2656017243862152], [0.09556570649147034, 0.06895843148231506, 0.4857999086380005, 0.1649225652217865], [0.20295970141887665, 0.03882686793804169, 0.0032911300659179688, 0.18110018968582153]], dtype='float32').reshape([5, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7de98865095104586a36bb62cf1bb2f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.17222259938716888, 0.0213778018951416, 0.0815553367137909, 0.4708496928215027], [0.3400913178920746, 0.10350854694843292, 0.17224067449569702, 0.20482248067855835], [0.11856469511985779, 0.3047958016395569, 0.4490484595298767, 0.023579567670822144], [0.07703478634357452, 0.2745734453201294, 0.10718491673469543, 0.17046567797660828]], dtype='float32').reshape([4, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f9a4c3063a69317bfbe8e696a88ca423(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([2055, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a3c63e618d497241ad18c2ae7979759a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3427695035934448, 0.28188279271125793, 0.04130434989929199, 0.07208472490310669], [0.3427695035934448, 0.28188279271125793, 0.04130434989929199, 0.07208472490310669], [0.4090425968170166, 0.09584981203079224, 0.005407147109508514, 0.2912749648094177], [0.21469947695732117, 0.27537813782691956, 0.03893736004829407, 0.1820843368768692], [0.09681469202041626, 0.029102057218551636, 0.02650250494480133, 0.31715601682662964], [0.22807008028030396, 0.12321293354034424, 0.10655063390731812, 0.09936696290969849], [0.1451793611049652, 0.09204316884279251, 0.3398480713367462, 0.12812256813049316]], dtype='float32').reshape([7, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_10603b033e704c20a82a1615acc47873(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_245dec642cf7c348e9e10f54d7404e35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([4838], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3a64bf269581d1fc5338d6a505b43dc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([1203], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_15d82534cfe0649debe2f9f69592ad42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([1045, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ae8170cc68487b63ae04e389a75be4f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.04082033038139343, 0.0395827442407608, 0.049334749579429626, 0.44707754254341125], [0.1399938017129898, 0.10986115038394928, 0.24484361708164215, 0.33283671736717224], [0.1399938017129898, 0.10986115038394928, 0.24484361708164215, 0.33283671736717224], [0.2036714255809784, 0.3061131536960602, 0.0289955735206604, 0.179578959941864], [0.3312148451805115, 0.01572442054748535, 0.25760287046432495, 0.05200636386871338], [0.03958702087402344, 0.20359259843826294, 0.0037869513034820557, 0.09604786336421967]], dtype='float32').reshape([6, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c3ec058bd8dc19ecb941c9ef740972ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([2335, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fcadba72a7a612d62e7ec1d0f833184a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([2983, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0a8a0b4ac7b65641c99eb69272266b4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([3778, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e1be1644f864c43c9c29bd4eaf078bb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.to_tensor([0.2580372095108032, 0.008511911146342754, 0.08306476473808289, 0.2119271457195282, 0.17736418545246124, 0.23127785325050354, 0.16143758594989777, 0.11461933702230453, 0.08355482667684555, 0.023082055151462555, 0.018276918679475784, 0.06076772138476372, 0.08422767370939255, 0.05787094309926033, 0.17422573268413544, 0.2186773419380188, 0.2362399846315384, 0.14056776463985443, 0.1850333958864212, 0.11165334284305573], dtype='float32').reshape([20]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1050b2bed6e7164e163c5896c18b630d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73d667a27d5bd50ac9d7a926952b5f8
    def get_inputs(self):
        return [
            paddle.uniform([17313], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_afb3bcc7fc3855f2ca087c292f1033c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([2064, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ed9a81ccae85e0e0ace330580c76bb63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.11357393860816956, 0.026668697595596313, 0.18120484054088593, 0.11391028761863708], [0.1228940486907959, 0.20003452897071838, 0.1861112415790558, 0.029059715569019318], [0.1525183916091919, 0.00827115774154663, 0.10151204466819763, 0.01746976375579834], [0.1525183916091919, 0.00827115774154663, 0.10151204466819763, 0.01746976375579834], [0.0808153748512268, 0.08832547068595886, 0.03856772184371948, 0.12868009507656097]], dtype='float32').reshape([5, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d2538e4c2f5899e0f43d9ff2bd8c840b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.uniform([4223, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_02d715da7e603518e92d2935bf79d364(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67e19920f71a01a6ea4974d6ddab58cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06543698161840439, 0.19422008097171783, 0.1002606600522995, 0.33282941579818726], [0.0316791832447052, 0.3710459768772125, 0.35150691866874695, 0.06539557874202728], [0.2985777258872986, 0.10518863797187805, 0.04721434414386749, 0.2629307508468628], [0.06543698161840439, 0.19422008097171783, 0.1002606600522995, 0.33282941579818726], [0.03976708650588989, 0.19435787200927734, 0.326302707195282, 0.2337922602891922], [0.20125213265419006, 0.15756282210350037, 0.09109322726726532, 0.31366896629333496], [0.03976708650588989, 0.19435787200927734, 0.326302707195282, 0.2337922602891922]], dtype='float32').reshape([7, 4]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
class TestPrimitiveOp_9f7aa7a6931dbd1de7769240b99cc789(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([1812, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8fd651b55d90be5831447fb8f42bc75d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([5519, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4098110eff4c54520ed69d4d33e1960b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([1798, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d80620d90f8e40ac4ef5493467f15758(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([1574, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9fbeb574dd19dd2a783150fd7c9ae7d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([2055, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_07182a42bc3ca50750fa7d3d715f8b75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_11e9f5483322a11a2aefc4b4ee54d289(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([1045, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_222b0ca4171e02b86569c363cc6fcdb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([2335, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a606d81c66064de9a5baaf86bacc8a64(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([2983, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bb9b9829b0b78a60f4a249dba0b335aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([3778, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0b95d206d35af039bc7cabde026623e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([2064, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
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
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9877d1a086ceeb845fd547a704fb2195(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27c306d248877430ed152c9699bd02ec
    def get_inputs(self):
        return [
            paddle.uniform([4223, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                if next(counter) == 0:
                    panic_stderr = f"stderr: \n{try_run_stderr}"
                else:
                    panic_stderr = "panic stderr have been reported by the first test case."
                raise RuntimeError(f"panicked. {panic_stderr}")
        return self._test_entry()


if __name__ == '__main__':
    unittest.main()