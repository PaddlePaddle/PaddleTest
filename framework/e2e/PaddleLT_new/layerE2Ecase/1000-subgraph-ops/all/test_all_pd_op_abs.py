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
class PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_42f3e819c2d7cc4a96447324414c90a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9bd6098584f77042aca9340fb757dedc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1acec7ee93cc6ead1ac09671b0909e43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
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

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2ad099c5b00eec955c8d4ab82fd7800c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
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
class TestPrimitiveOp_036161ebc942738b8e9ce5f7fbaeea69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([1812, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_e8cf833bcadec6d72b379d99c91af4a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.030847802758216858, -0.1512724906206131, 0.39145582914352417, 0.0012667477130889893], [0.27543580532073975, -0.07220971584320068, 0.3511688709259033, 0.18944287300109863], [-0.3136575520038605, -0.07451048493385315, 0.2333432137966156, -0.34193605184555054], [-0.3953903913497925, 0.09340949356555939, 0.13210399448871613, -0.20949581265449524], [0.13511842489242554, -0.20150819420814514, 0.3850752115249634, 0.11844278872013092]], dtype='float32').reshape([5, 4]),
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
class TestPrimitiveOp_244f4be19a1d2cacf2769bd21ec663a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.23857006430625916, -0.2494465708732605, 0.26256534457206726, 0.11029650270938873], [-0.15996772050857544, -0.02149178832769394, 0.1710740625858307, 0.1583121120929718], [-0.03202125057578087, -0.009222090244293213, 0.2052132934331894, 0.2637884318828583], [-0.15996772050857544, -0.02149178832769394, 0.1710740625858307, 0.1583121120929718], [-0.03202125057578087, -0.009222090244293213, 0.2052132934331894, 0.2637884318828583]], dtype='float32').reshape([5, 4]),
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

class PrimitiveOp_a4ff3fffb2f87a97e46dcd79001ac905(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8947a642679c7ddf12a15b90d5656a86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4ff3fffb2f87a97e46dcd79001ac905
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b3ccfdaf52a2e9e094dd29d218eb8058(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([5519, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_06e0e4976157c509f5c77a8af843a3aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3680111765861511, 0.01847441866993904, 0.05460155010223389, -0.015949957072734833], [-0.26094627380371094, -0.1204889714717865, -0.2633415460586548, 0.11405646800994873], [-0.1106850802898407, -0.43407687544822693, 0.04854273796081543, -0.12346945703029633], [-0.26094627380371094, -0.1204889714717865, -0.2633415460586548, 0.11405646800994873], [-0.1106850802898407, -0.43407687544822693, 0.04854273796081543, -0.12346945703029633], [-0.07249720394611359, 0.2051759958267212, 0.12682610750198364, 0.1690688580274582], [-0.07249720394611359, 0.2051759958267212, 0.12682610750198364, 0.1690688580274582]], dtype='float32').reshape([7, 4]),
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
class TestPrimitiveOp_317505ba8c6aef334d447484b9a51c66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_359d4df2bf5a9a558a2ae40222558ed2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
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
class TestPrimitiveOp_e116405745ed6ac7eea409112fced8a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([1798, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_234f58567a945de3b80dab2a8157a7f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([1574, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ea88602b7560e761dfc46019f9dc3930(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.14555048942565918, 0.005882769823074341, -0.11834472417831421, 0.1643039435148239], [0.2269618809223175, 0.2803482413291931, 0.0850462019443512, -0.1952138990163803], [-0.024031832814216614, 0.08829766511917114, 0.10472142696380615, 0.012247443199157715], [0.23522406816482544, 0.149470716714859, 0.23910215497016907, -0.02997654676437378], [0.23522406816482544, 0.149470716714859, 0.23910215497016907, -0.02997654676437378], [-0.024031832814216614, 0.08829766511917114, 0.10472142696380615, 0.012247443199157715]], dtype='float32').reshape([6, 4]),
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
class TestPrimitiveOp_3d55ca35acaeab375e6c62e5211e2d04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.20295970141887665, -0.03882686793804169, -0.0032911300659179688, -0.18110018968582153], [-0.10731416940689087, -0.11742308735847473, -0.39797326922416687, -0.13853870332241058], [0.0022506117820739746, -0.3054242432117462, 0.15239760279655457, -0.2656017243862152], [0.09556570649147034, -0.06895843148231506, -0.4857999086380005, -0.1649225652217865], [0.20295970141887665, -0.03882686793804169, -0.0032911300659179688, -0.18110018968582153]], dtype='float32').reshape([5, 4]),
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
class TestPrimitiveOp_1c3cdca0ecbd944d6052c253aa10e6de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
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
class TestPrimitiveOp_3ff7e581135ba9242127b231d3bd21ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.17222259938716888, -0.0213778018951416, -0.0815553367137909, -0.4708496928215027], [0.3400913178920746, 0.10350854694843292, 0.17224067449569702, 0.20482248067855835], [-0.11856469511985779, -0.3047958016395569, 0.4490484595298767, -0.023579567670822144], [-0.07703478634357452, 0.2745734453201294, 0.10718491673469543, -0.17046567797660828]], dtype='float32').reshape([4, 4]),
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
class TestPrimitiveOp_b29ae46566881c74026cbc33c81b657f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
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
class TestPrimitiveOp_01c4a4c9683f44e735e8cbafa2015b4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([2055, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8b096b34feece965ede8b5ae946132b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.3427695035934448, -0.28188279271125793, 0.04130434989929199, 0.07208472490310669], [0.3427695035934448, -0.28188279271125793, 0.04130434989929199, 0.07208472490310669], [-0.4090425968170166, -0.09584981203079224, 0.005407147109508514, -0.2912749648094177], [-0.21469947695732117, 0.27537813782691956, -0.03893736004829407, 0.1820843368768692], [0.09681469202041626, 0.029102057218551636, 0.02650250494480133, -0.31715601682662964], [-0.22807008028030396, 0.12321293354034424, -0.10655063390731812, -0.09936696290969849], [-0.1451793611049652, -0.09204316884279251, -0.3398480713367462, -0.12812256813049316]], dtype='float32').reshape([7, 4]),
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
class TestPrimitiveOp_d7498214574dd72d374685393fb70189(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d0fd2624cb61b35704a9373b7d3aaeda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4ff3fffb2f87a97e46dcd79001ac905
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_23e948826b430f5c16202d198456caa1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([4590, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_2b161a220d66a5d98e49bae8208e4a1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([1045, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5f7557ec52938aac398955c93c759e14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.04082033038139343, 0.0395827442407608, -0.049334749579429626, -0.44707754254341125], [-0.1399938017129898, -0.10986115038394928, -0.24484361708164215, -0.33283671736717224], [-0.1399938017129898, -0.10986115038394928, -0.24484361708164215, -0.33283671736717224], [0.2036714255809784, 0.3061131536960602, -0.0289955735206604, -0.179578959941864], [-0.3312148451805115, -0.01572442054748535, 0.25760287046432495, 0.05200636386871338], [-0.03958702087402344, -0.20359259843826294, -0.0037869513034820557, 0.09604786336421967]], dtype='float32').reshape([6, 4]),
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

class PrimitiveOp_140781b5a63852c07669f83ad482e5b9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_77939128bfdb23a71cc043004251dde3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_140781b5a63852c07669f83ad482e5b9
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
class TestPrimitiveOp_e0f4e278d6cfc0e59815bbcca73f0bbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_140781b5a63852c07669f83ad482e5b9
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
class TestPrimitiveOp_4315fa6a82073fa03ecc8b7a1d28f962(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4ff3fffb2f87a97e46dcd79001ac905
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_887cd84f693eb98c017ed23a4b14fa83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([2335, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_61852fc4efa16f9d7759d79d756fedb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([2983, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_0cbf4883787a8ffa66639d206cab0a9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([3778, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_45b3fe7313c9f328a2f7430ed9fef981(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4ff3fffb2f87a97e46dcd79001ac905
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_7c05f754a9b7102fde118c7ad7035bc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_deb84d337d6dfe17198c581d2ee653b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
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
class TestPrimitiveOp_8168fc5742243cc66a28a878b063b3e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([2064, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_862ad5c08c96fe27bda5e83be53b59b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4ff3fffb2f87a97e46dcd79001ac905
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_2dc9d3e5ed9591c79d73e136188a9c55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.11357393860816956, 0.026668697595596313, 0.18120484054088593, -0.11391028761863708], [0.1228940486907959, 0.20003452897071838, 0.1861112415790558, -0.029059715569019318], [-0.1525183916091919, -0.00827115774154663, 0.10151204466819763, -0.01746976375579834], [-0.1525183916091919, -0.00827115774154663, 0.10151204466819763, -0.01746976375579834], [-0.0808153748512268, 0.08832547068595886, -0.03856772184371948, 0.12868009507656097]], dtype='float32').reshape([5, 4]),
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
class TestPrimitiveOp_19a50f2ef779b6a824d8dc50973f5279(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
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
class TestPrimitiveOp_e2c6e6c0ffa730cb12fa2f7b26f81b62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([4223, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_86aa5fc06502f8c1c10aea7f8bda41bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.06543698161840439, -0.19422008097171783, 0.1002606600522995, 0.33282941579818726], [0.0316791832447052, -0.3710459768772125, 0.35150691866874695, 0.06539557874202728], [-0.2985777258872986, -0.10518863797187805, 0.04721434414386749, -0.2629307508468628], [0.06543698161840439, -0.19422008097171783, 0.1002606600522995, 0.33282941579818726], [-0.03976708650588989, -0.19435787200927734, 0.326302707195282, 0.2337922602891922], [0.20125213265419006, -0.15756282210350037, -0.09109322726726532, -0.31366896629333496], [-0.03976708650588989, -0.19435787200927734, 0.326302707195282, 0.2337922602891922]], dtype='float32').reshape([7, 4]),
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
class TestPrimitiveOp_cdf13b1808d8b2c37694594accad3bb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
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

class PrimitiveOp_d926397ea49ebaec008021a33053e131(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d35159c812136c410c04f58e993d33aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d926397ea49ebaec008021a33053e131
    def get_inputs(self):
        return [
            paddle.uniform([1812, 4], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_f094763ca7b43d3df23bff5f0564d9d1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7d23068e88b59c473c1d55c166e13047(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f094763ca7b43d3df23bff5f0564d9d1
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_3f238d77ea3fe60915ed81148fcfc107(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d926397ea49ebaec008021a33053e131
    def get_inputs(self):
        return [
            paddle.uniform([5519, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_45c6e97747e0c7ea9c96a739cdacc596(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d926397ea49ebaec008021a33053e131
    def get_inputs(self):
        return [
            paddle.uniform([1798, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4dde9bf497298b3328dc98f4e87b8d3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d926397ea49ebaec008021a33053e131
    def get_inputs(self):
        return [
            paddle.uniform([1574, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_de463a3176b666b61ffed74b571d83ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d926397ea49ebaec008021a33053e131
    def get_inputs(self):
        return [
            paddle.uniform([2055, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_e6ec971f06fb20c7dffe0359eba1969e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f094763ca7b43d3df23bff5f0564d9d1
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1c6e505922968598d47868fdd61d8e46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d926397ea49ebaec008021a33053e131
    def get_inputs(self):
        return [
            paddle.uniform([4590, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d4d2c43872209d8a312a92964c00b77a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d926397ea49ebaec008021a33053e131
    def get_inputs(self):
        return [
            paddle.uniform([1045, 4], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_0aabeb1a1c7f4fc3cf2c22504bc24db5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6ccbab02660e2e6c435c37c926c03850(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0aabeb1a1c7f4fc3cf2c22504bc24db5
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

class PrimitiveOp_a9091857dccdd48b1f987aa8f880da63(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_567bdff3912809446bd1ca945cb328a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9091857dccdd48b1f987aa8f880da63
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
class TestPrimitiveOp_149512f814521e6fa5240c63cee3cd20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f094763ca7b43d3df23bff5f0564d9d1
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9b04050601fa25d071eaacbb50ec8a91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d926397ea49ebaec008021a33053e131
    def get_inputs(self):
        return [
            paddle.uniform([2335, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_027c77487187b68102245f1fb2c28ff0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d926397ea49ebaec008021a33053e131
    def get_inputs(self):
        return [
            paddle.uniform([2983, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c3dd8f3652ccb279740df87d184adde5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d926397ea49ebaec008021a33053e131
    def get_inputs(self):
        return [
            paddle.uniform([3778, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ccfa6811d4827c1d783ec4b0bfe950b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f094763ca7b43d3df23bff5f0564d9d1
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9e222952514cecd242318f158b642d55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d926397ea49ebaec008021a33053e131
    def get_inputs(self):
        return [
            paddle.uniform([2064, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9f80e86d66e22f979bf757a27f34e3b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f094763ca7b43d3df23bff5f0564d9d1
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_460bd7585ec28b0bf01a5d58a60da642(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d926397ea49ebaec008021a33053e131
    def get_inputs(self):
        return [
            paddle.uniform([4223, 4], dtype='float32', min=0, max=0.5),
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