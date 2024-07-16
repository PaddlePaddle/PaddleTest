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
class PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_092b65dbd396b4d1c62e42527ab61109(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_22381fbd1daa84325ea8679549f1bef6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_92609cc858501bdb718e973191477022(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.to_tensor([[4.685884952545166, 4.341721534729004, 5.0854668617248535, 4.807162284851074, 4.381277561187744, 4.854126453399658, 4.59234094619751, 4.803297519683838, 4.440852642059326, 4.615706443786621, 3.8427395820617676, 5.17818546295166, 4.46326208114624, 4.0751471519470215, 4.623128890991211, 4.63656759262085, 5.268134117126465, 4.743747711181641]], dtype='float32').reshape([1, 18]),
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
class TestPrimitiveOp_2dbdade82fe2bf88affab0e8dc434dac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.to_tensor([[6.394744873046875, 5.274114608764648, 5.703514099121094, 6.524235248565674, 5.836944103240967, 6.776856422424316, 6.140979766845703, 6.095730781555176, 5.962957859039307, 6.768327236175537, 6.455108642578125, 6.547737121582031, 5.934031009674072, 5.640436172485352, 6.510316371917725, 5.865507125854492, 5.536918640136719, 5.9943742752075195, 5.177901744842529, 6.1067633628845215, 6.230833530426025, 5.81798791885376, 6.211367130279541]], dtype='float32').reshape([1, 23]),
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
class TestPrimitiveOp_3401804b02a29cd50a5bcb4b53aaa1da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_446a456a3b619b11e7fb755b90848e51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([1, 240], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_57d7e1cd55a10db6e90a4fad0163e48e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5267e2e559495404adee770c1be62abc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 20, 32], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_bb8fba8be6aec9eb71462bd27ea36540(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_879465da1871797ec9c6362b8dd1b4b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_e417fd112f57feff6f5176bd5a5ea10a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.709053039550781]], [[7.550483703613281]], [[7.344411849975586]], [[6.536085605621338]], [[7.039064407348633]], [[6.679584503173828]], [[7.219615936279297]], [[7.443569660186768]], [[7.062199115753174]], [[6.990291118621826]], [[7.615472793579102]], [[6.663925647735596]], [[6.661057949066162]], [[7.0189337730407715]], [[7.696951866149902]], [[7.372894763946533]], [[7.589016914367676]], [[7.98672342300415]], [[7.5524468421936035]], [[7.484039783477783]], [[7.152288913726807]], [[7.274065971374512]], [[7.17875862121582]], [[6.769964218139648]], [[7.289031028747559]], [[7.488609313964844]], [[7.0675153732299805]], [[7.440677642822266]], [[7.4233222007751465]], [[6.610384464263916]]]], dtype='float32').reshape([1, 30, 1, 1]),
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
class TestPrimitiveOp_464af458cccfa94e9dc8d0edcaa1beb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f835c166a465ca503332b1fe10d73fdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b7ba0dcdecdaa4d18a1ed19338851a9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_46327ce353f73c04ccdb95db1c22f76f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f96bc62e6e16b65dbf1a66098109b82a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_0e66645fb520c750261ae5db24f7d529(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_0a0dc1b0109a3250c9f247940d83af48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.439476013183594]], [[7.116119861602783]], [[7.933470726013184]], [[7.429988384246826]], [[7.739742755889893]], [[7.160177230834961]], [[6.873104572296143]], [[6.83212423324585]], [[8.443646430969238]], [[7.169576644897461]], [[7.918055534362793]], [[8.18014144897461]], [[7.301506519317627]], [[7.215599060058594]], [[7.0682148933410645]], [[7.000136375427246]], [[8.593505859375]], [[6.998315334320068]], [[7.606921672821045]], [[7.383068084716797]], [[7.5837273597717285]], [[7.888897895812988]], [[7.603029251098633]], [[8.677313804626465]], [[8.388718605041504]], [[7.459686756134033]], [[8.030505180358887]], [[7.010253429412842]], [[8.303638458251953]], [[7.043381214141846]]]], dtype='float32').reshape([1, 30, 1, 1]),
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
class TestPrimitiveOp_96b462f30c39c94a210b4774af293322(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 50, 76], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f996eccffff56a116b45332db39e3bac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.5144989490509033]], [[1.4872188568115234]], [[1.3116577863693237]], [[1.1540439128875732]], [[1.335241675376892]]]], dtype='float32').reshape([1, 5, 1, 1]),
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
class TestPrimitiveOp_b176077d513b397ffa9b52acaa0f58ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.831326484680176]], [[2.8996922969818115]], [[2.341322422027588]], [[2.725881338119507]], [[2.1794872283935547]], [[2.3146777153015137]], [[3.153815269470215]], [[2.625821352005005]], [[2.693358898162842]], [[3.4684395790100098]]]], dtype='float32').reshape([1, 10, 1, 1]),
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
class TestPrimitiveOp_42d3f785a68f35ddaefc9270daf0c347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_84888c9a41ca15b2ed2ea0d5b4ee02f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.052102088928223]], [[5.973732948303223]], [[6.436169624328613]], [[5.944063663482666]], [[6.007678508758545]], [[6.059599876403809]], [[5.8827714920043945]], [[6.284278869628906]], [[6.138669013977051]], [[5.883573055267334]], [[5.338260173797607]], [[5.5233683586120605]], [[5.554606914520264]], [[5.810969352722168]], [[5.240361213684082]], [[6.271805286407471]], [[6.381803512573242]], [[6.284727096557617]], [[6.23016357421875]], [[5.052532196044922]], [[5.897260665893555]], [[5.679372310638428]], [[5.945131301879883]], [[6.913031101226807]]]], dtype='float32').reshape([1, 24, 1, 1]),
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
class TestPrimitiveOp_640dfc096a2dd3353d5e664045fb8bfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 100, 152], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_726796571fb0bd926f858b5e99776c68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 13, 19], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d1546bc82fe2eb799580686b3e84e4f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([10, 15], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c0e8cd40b00e183864b3343efa31845c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.624106407165527]], [[4.701679706573486]], [[3.785703659057617]], [[4.636858940124512]], [[4.185674667358398]], [[5.190023422241211]], [[3.895230293273926]], [[3.9510366916656494]], [[4.272900581359863]], [[4.303114891052246]], [[4.5292229652404785]], [[3.799717664718628]], [[4.855134963989258]], [[4.441648006439209]], [[4.650888442993164]], [[4.795456886291504]], [[4.847243309020996]], [[4.663857460021973]]]], dtype='float32').reshape([1, 18, 1, 1]),
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
class TestPrimitiveOp_0d5e3ab9ea0419beecfdd7350cb863c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.78853702545166]], [[7.146003246307373]], [[6.132681846618652]], [[6.221840858459473]], [[6.431774616241455]], [[6.59204626083374]], [[6.623891353607178]], [[6.397099494934082]], [[6.569820880889893]], [[7.090774059295654]], [[5.756249904632568]], [[5.597261905670166]], [[5.9224958419799805]], [[5.9571452140808105]], [[5.673080921173096]], [[6.735020637512207]], [[6.120677947998047]], [[6.651705741882324]], [[6.907902717590332]], [[6.890196800231934]], [[6.471126079559326]], [[6.618170261383057]], [[5.1616950035095215]], [[7.4771199226379395]]]], dtype='float32').reshape([1, 24, 1, 1]),
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
class TestPrimitiveOp_0ff1bb038b65a20fb32ac2296db3f953(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([145, 84], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f023a7523dc66aaf5816fa4875f37c8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 28, 40], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_84b441d19110890ed512672fccb4195c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.544837236404419]], [[0.8169614672660828]], [[1.2499406337738037]], [[1.3618935346603394]]]], dtype='float32').reshape([1, 4, 1, 1]),
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
class TestPrimitiveOp_223a72dc93d92cb60bd80a3d2c9a708f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.406261444091797]], [[2.7747883796691895]], [[3.009711980819702]], [[3.176870346069336]], [[2.6877174377441406]], [[3.1237170696258545]], [[2.2043473720550537]], [[2.562354564666748]], [[3.4220175743103027]], [[3.2416279315948486]], [[2.9616148471832275]]]], dtype='float32').reshape([1, 11, 1, 1]),
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
class TestPrimitiveOp_47c034594326f8389515e0716c592668(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_75113dae5b67bcc7effc8e3494104f42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.471386909484863]], [[7.349374771118164]], [[7.80717134475708]], [[6.732783794403076]], [[7.302953720092773]], [[7.446574687957764]], [[8.105687141418457]], [[8.2003812789917]], [[7.494767189025879]], [[7.934525966644287]], [[7.961569786071777]], [[6.840730667114258]], [[7.564905166625977]], [[7.182055473327637]], [[7.827128887176514]], [[7.650530815124512]], [[7.258990287780762]], [[7.523951053619385]], [[7.0820393562316895]], [[6.465030193328857]], [[6.736588478088379]], [[7.990978240966797]], [[7.304441452026367]], [[7.74916934967041]], [[6.747164249420166]], [[8.26745891571045]], [[7.089442729949951]], [[7.226877212524414]], [[7.833673000335693]], [[7.621578693389893]]]], dtype='float32').reshape([1, 30, 1, 1]),
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
class TestPrimitiveOp_f0624971cbd4a881c7610f16101a9101(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c0a9f68967862ba44e17d58f2f6ac06c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 80, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_e8aaf848a9b83da524ff17ca9ec07e59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.8788163661956787]], [[4.8822431564331055]], [[4.714454174041748]], [[5.47055721282959]], [[5.005043029785156]], [[4.7693023681640625]], [[4.298551559448242]], [[4.079020023345947]], [[4.330775737762451]], [[4.043824195861816]], [[4.479321479797363]], [[4.7139410972595215]], [[4.095968246459961]], [[5.0010576248168945]], [[4.4950785636901855]], [[4.312562942504883]]]], dtype='float32').reshape([1, 16, 1, 1]),
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
class TestPrimitiveOp_fb64d54ab6dbdad0f38aff15c92b1dd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 14, 20], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8d10dfd8390172a6d33e401a50d8e008(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 22, 33], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_01f2dd3158c1d89d2366eeafbeab9a2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 23, 35], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_89baffe015f7216d4cdce9b5c9c10dfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 46, 70], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_0e91b73c7d96c894da88047aa730444a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([22, 15], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_31d2f46305c95cf7b097b001043f9d00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_78c0e58b83882427b29ac42997283380(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.351052284240723]], [[9.076446533203125]], [[7.611941337585449]], [[9.164783477783203]], [[7.442530155181885]], [[8.784290313720703]], [[8.335529327392578]], [[8.586498260498047]], [[8.109635353088379]], [[7.509426116943359]], [[8.61800479888916]], [[8.74172306060791]], [[7.5675435066223145]], [[8.286055564880371]], [[8.676008224487305]], [[8.710409164428711]], [[8.41573715209961]], [[8.553010940551758]], [[9.207554817199707]], [[8.445959091186523]], [[8.4314603805542]], [[9.244460105895996]], [[8.93134593963623]], [[7.5370073318481445]], [[8.284849166870117]], [[8.928435325622559]], [[9.228302001953125]], [[7.232161998748779]], [[8.493978500366211]], [[8.002117156982422]]]], dtype='float32').reshape([1, 30, 1, 1]),
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
class TestPrimitiveOp_efa9795366e794b5f707ee2333924781(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_586ef4bd6caa9ffe584fd3d46d57e1ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([1, 218], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_0d91d4e5ac555513320b63f7fb5f3162(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.523366451263428]], [[6.616939544677734]], [[6.126245021820068]], [[5.82805061340332]], [[6.883699417114258]], [[7.409236907958984]], [[6.320303916931152]], [[6.522290229797363]], [[6.37861442565918]], [[6.731780529022217]], [[6.4535746574401855]], [[6.217649936676025]], [[5.953060150146484]], [[5.921178340911865]], [[7.176547050476074]], [[7.22666072845459]], [[7.4815850257873535]], [[6.360232353210449]], [[6.470932483673096]], [[6.494004726409912]], [[6.564431190490723]], [[6.752301216125488]], [[6.501801013946533]], [[6.448027610778809]], [[6.599648475646973]]]], dtype='float32').reshape([1, 25, 1, 1]),
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
class TestPrimitiveOp_2a45bfa5bf36752434cbbf0790ed4147(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 6, 9], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d14876fbacc008f768bef877e95bf35a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_da6afc03045d49075bb50fe89d743e85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([390, 1024], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ff690b20520e7b747adf723f1f300324(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([171, 84], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_3e38b5168b78349770973007d1bb595b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_14a88ded18575387d39fb968349d4aab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.823836326599121]], [[4.841972351074219]], [[4.305678844451904]], [[4.914802551269531]], [[5.130475044250488]], [[4.788827896118164]], [[5.480341911315918]], [[5.343693733215332]], [[5.133093357086182]], [[5.910193920135498]], [[5.054744243621826]], [[4.827784538269043]], [[4.356714248657227]], [[4.685656547546387]], [[5.108277320861816]], [[5.094736099243164]], [[5.665359020233154]], [[5.948004245758057]], [[5.116445541381836]], [[4.770357608795166]]]], dtype='float32').reshape([1, 20, 1, 1]),
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
class TestPrimitiveOp_94445dd2a983f8ebf1c5c02de95985c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.93044900894165]], [[4.609947204589844]], [[5.418363094329834]], [[5.954451084136963]], [[4.70367431640625]], [[4.194159984588623]], [[5.230362415313721]], [[4.139449596405029]], [[5.085805892944336]], [[4.790792465209961]], [[5.196610450744629]], [[4.634404182434082]], [[4.593898296356201]], [[5.541975498199463]], [[5.049737930297852]], [[4.953497886657715]], [[4.082186698913574]], [[4.560928821563721]]]], dtype='float32').reshape([1, 18, 1, 1]),
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
class TestPrimitiveOp_a152b3e389f83c25129f7fef7e6d16b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_74f91af3ca094b6bd1f221bb488d6364(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 7, 10], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ba28d2c0d2ac305cb330bcbab2bee283(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 109, 109], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c1885f7df7c99c8b6a3abce19722b2be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 54, 54], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ed1ab301c7e5ffa98dd8bf76917d9c80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a343f8c4dfa539fabe2d3f4e551b11a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 54, 54], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b296e67c7a15989a0401e9a82248bf32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 54, 54], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6e3f22cabcaae2c2246eafd5d9f82cdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8be4483914f541e97131197a04587f6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_880ae0f2b67ca60abd3392037be6c682(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_990bb21634197b53726efbdb8898454c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5018383c75c1f9f715ca35ac187a3d20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_75d0ec8403dc2514ed8b67917399f8fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_e7cff64261df91950649e72aedce25f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_68179bbda00ae25bb61e917b6fed0803(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_613d3999537851ef53f7cbd0a60f47f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([43, 1000, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_379362da6a6abc8e3095b799d3d2613e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.953296184539795]], [[4.5094313621521]], [[4.1247944831848145]], [[5.052865982055664]], [[4.30315637588501]], [[4.862283706665039]], [[3.980379104614258]], [[4.318688869476318]], [[4.388061046600342]], [[4.830044269561768]], [[4.465309143066406]], [[4.359241008758545]], [[4.2101731300354]], [[4.413375377655029]], [[4.822516441345215]], [[4.36255407333374]], [[4.419497489929199]], [[4.613299369812012]]]], dtype='float32').reshape([1, 18, 1, 1]),
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
class TestPrimitiveOp_180b6cc5753a854275b9d2f722040038(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.908949851989746]], [[7.10638952255249]], [[5.833196640014648]], [[7.675716876983643]], [[6.564254283905029]], [[7.208909034729004]], [[7.60159158706665]], [[6.765165328979492]], [[7.721771240234375]], [[6.488300323486328]], [[6.992899417877197]], [[7.120333194732666]], [[5.930432319641113]], [[7.055731773376465]], [[6.70198917388916]], [[6.966367721557617]], [[5.794203758239746]], [[6.519737243652344]], [[6.744708061218262]], [[6.58433723449707]], [[6.722794532775879]], [[6.882261276245117]], [[6.998544216156006]], [[6.971408367156982]]]], dtype='float32').reshape([1, 24, 1, 1]),
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
class TestPrimitiveOp_e0e92ed3c15ab33ebe2736a118074401(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 11, 17], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_aee1652febf84017a9d72c7c8146b4bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.223539352416992]], [[4.981096267700195]], [[4.280937671661377]], [[5.172879695892334]], [[5.1251115798950195]], [[4.71785306930542]], [[5.362261772155762]], [[5.090798377990723]], [[5.013430595397949]], [[4.217739105224609]], [[4.987013339996338]], [[4.386631965637207]], [[5.021262168884277]], [[4.720359802246094]], [[5.020998954772949]], [[5.547306537628174]], [[5.341071128845215]], [[4.673004150390625]]]], dtype='float32').reshape([1, 18, 1, 1]),
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
class TestPrimitiveOp_e1e2211dbc8077db890c66d7c8ff8621(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d51cbff9a60e39c1793c385646392ba8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 10, 16], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d41d869f763da8dc86d26a319998738a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.273597240447998]], [[3.9308645725250244]], [[4.461357116699219]], [[4.238923072814941]], [[4.377993583679199]], [[4.115373611450195]], [[4.863597393035889]], [[4.693233966827393]], [[4.529014587402344]], [[4.364161968231201]], [[4.21715784072876]], [[4.133312225341797]], [[4.583900451660156]], [[4.583021640777588]], [[4.651145935058594]], [[4.6100592613220215]], [[4.974761009216309]], [[4.1594977378845215]]]], dtype='float32').reshape([1, 18, 1, 1]),
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
class TestPrimitiveOp_b1b84b0409832ad89d473ce72cba4c30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([10, 9], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_33373fb20dec1eda3b91e69b77f22a2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 12, 18], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_3bdff070ac598df08ac5bfa448fa0a2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([10, 96, 109, 109], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a78536f22d78f63de7ffe63015750ed1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 54, 54], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_713b381b0bfc4f54ff416c207586f86c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_56b29588d4c17bca909026df84e57038(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 54, 54], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b930e4879c4318d4c7fe7330ea9221b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 54, 54], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ca8b152cb7a7e7508e3c4f4b73bf3d73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ac981de4b8dfac05d0d2abce52a8fdd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6a8cf1536dad269fdf08ba10cad4655d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([10, 48, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_af0e594c7628ada57802c0427118cce2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f7257c7df4d94908e2f283414a7ea6e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_bf3dfc409429dbf5976ee98cd5dc95e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_7bb6c31c46614653f7ee710542887353(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f823f7ba36751fc32f7d551e6a7ee895(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_2b07d9ac2f242b99f910788f50bae5ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([10, 1000, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ccf5a336767d84122d5ebb7e6708cb31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([10, 120], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b7356e4cf4a8843361dc1d4304aa49bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([22, 84], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_55eec19f18de1200e2b32bbfe14b16f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 92, 140], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1da40d4c126db9d264b6f934319ff57f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f2601f63dead1c9a2c0a5fbba57bc056(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_7990a0c5f16903155a351e64f9b4e294(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 300, 300], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_da883cdaf42311c466d1b5b460fefee0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 150, 150], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8aef182d1155509105c4e64b1a01089d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_0dcb5ba085782e2b5ae8736f3820782a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f41233d3d7aa34928d639f1a096edda0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_34391d4e4f8d6ce36348e326bc84d9a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 19, 19], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4d05029828517a64710f5a5aa290c534(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 19], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_cfff910cc5302d4d5efeb9b90e846daf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 10, 10], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9e33d734094bb3a4be64418f674d806c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 10, 10], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1c48b2ad7441b4fe514083d71befd893(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 5], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6d12cd267519a1a3e278745ec3bc5330(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 5, 5], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c77a00164171a8236ba259fcbb7ad3b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_89678cb65773735651f55ab0e0fff433(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 3, 3], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ed020ae26d2872204a8ef84f6d35817f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ea2e767e1b4503e8c333d41327d9e973(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 13, 19], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_297f2a7018404b8e6d85d504c3485a7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([171, 15], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_78c41a3a3f2be954e260437f1e432417(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d4419ed6dbc2ddde534f614d66170b06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 25, 38], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_00e89e1c00581a3acf3691a99c08bad0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.113353729248047]], [[5.495994567871094]], [[5.71132755279541]], [[6.018656253814697]], [[4.675065994262695]], [[5.014891147613525]], [[4.965630054473877]], [[4.973498344421387]], [[6.757758140563965]], [[4.501132488250732]], [[5.846304893493652]], [[6.218576431274414]], [[5.281675338745117]], [[5.208498954772949]], [[5.148212432861328]], [[5.077774524688721]], [[5.3781046867370605]], [[4.8508477210998535]]]], dtype='float32').reshape([1, 18, 1, 1]),
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
class TestPrimitiveOp_8ef4fcbede536c985194725fa3c9a37d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 13, 19], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_309bcbdb0caa60528e4802a61354e730(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_937716d2f13c90333ea75b60b0afd4fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.057794570922852]], [[4.007081031799316]], [[4.410943031311035]], [[3.929718255996704]], [[4.576589584350586]], [[4.156850814819336]], [[4.325844764709473]], [[4.249763011932373]], [[4.070028305053711]], [[3.645435094833374]], [[3.646209239959717]], [[4.158534526824951]], [[4.211146354675293]], [[4.155292987823486]], [[4.120391368865967]], [[4.319928169250488]]]], dtype='float32').reshape([1, 16, 1, 1]),
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
class TestPrimitiveOp_a125ab3260ce80153d9b949a877b2872(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([22, 9], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c9379d130d7ba9d7a576c796b1ddcfaf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.6342549324035645]], [[4.274336338043213]], [[4.769364356994629]], [[5.671443939208984]], [[4.562361240386963]], [[5.569528102874756]], [[5.409242153167725]], [[5.292044639587402]], [[5.178565979003906]], [[5.39305305480957]], [[4.9184956550598145]], [[4.815407752990723]], [[5.106956481933594]], [[5.396690845489502]], [[4.686182975769043]], [[4.850823879241943]], [[5.380630016326904]], [[4.1114678382873535]]]], dtype='float32').reshape([1, 18, 1, 1]),
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
class TestPrimitiveOp_e693f106421878b884085000971c19ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.5894896984100342]], [[1.3004567623138428]], [[1.49582040309906]], [[1.4133695363998413]]]], dtype='float32').reshape([1, 4, 1, 1]),
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
class TestPrimitiveOp_5fe976d49df7abeeda3f8ba34644b404(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 109, 109], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_309ba78f54f0b6414c680144777b2b08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 54, 54], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_dae118162b003bbfe32044e2a3a181df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1015aa4401c089713afa8cb9756e1f90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 54, 54], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b5bea4b62e72ec355e7b36404bd75638(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 54, 54], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_43e59b5052dd516656bfdaa78308598f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_7801d44ca0095b3e6161b22db6b465ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c4e273e55aa578aedc5bb0e0cd6b6974(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5c479b777a317fda617482d930a73fff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d044510afbac988a587b29e3f3506f85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_74025c82d095c3c41987c40e8f7af9ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c2ea6540a0fdfde08dff061ba6b098d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_dcdd1314ded8dc52744122acea4bf802(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9b5cd3739909a438068995266dc5af48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([11, 1000, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_44abae1cba62ff965888cdd3dcc8075f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([145, 15], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_e431e684312e2fd7d42689b2089534d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([1, 168], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_32926fcd9fecccab88b4d097ddbc9b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f71b4d36fe3b73ba378bb27d2a76d3ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.70921516418457]], [[4.715009689331055]], [[5.236291408538818]], [[4.9092183113098145]], [[4.383486747741699]], [[4.950136184692383]], [[5.096542835235596]], [[4.879143714904785]], [[4.668126106262207]], [[5.038344383239746]], [[4.926565170288086]], [[5.196986198425293]], [[4.549531936645508]], [[4.688711643218994]], [[5.383689880371094]], [[5.174729824066162]], [[5.0076518058776855]], [[5.199132442474365]], [[4.6433563232421875]], [[4.807617664337158]]]], dtype='float32').reshape([1, 20, 1, 1]),
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
class TestPrimitiveOp_99ec7f20d2cde5b4ed74fe3c6545f778(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c61942a8a77f4ffdb583d67151c9ebf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.0434675216674805]], [[3.0076746940612793]], [[4.3080973625183105]], [[3.0539610385894775]], [[4.140078544616699]], [[3.647865056991577]], [[3.7724828720092773]], [[2.946160078048706]], [[2.7036635875701904]], [[2.934163808822632]], [[3.413623094558716]], [[3.533562183380127]]]], dtype='float32').reshape([1, 12, 1, 1]),
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
class TestPrimitiveOp_9482c8c4f1e94569c461708fbffa0b22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.164685249328613]], [[5.6693549156188965]], [[4.820473670959473]], [[5.086769104003906]], [[5.46857213973999]], [[4.941510200500488]], [[5.570308685302734]], [[5.455769062042236]], [[4.548212051391602]], [[5.573060035705566]], [[4.531099319458008]], [[5.059079647064209]], [[5.089148044586182]], [[5.488116264343262]], [[5.749452114105225]], [[4.732754707336426]], [[5.111810684204102]], [[5.471380233764648]], [[4.567051410675049]], [[5.362548828125]]]], dtype='float32').reshape([1, 20, 1, 1]),
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
class TestPrimitiveOp_60e369367755eb2f66934ce6aadb6864(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.554349660873413]], [[2.984191417694092]], [[3.0773959159851074]], [[2.9575600624084473]], [[2.2049384117126465]], [[3.0560779571533203]], [[2.0327553749084473]], [[3.163947343826294]], [[2.833745241165161]], [[2.7105236053466797]], [[1.7433278560638428]]]], dtype='float32').reshape([1, 11, 1, 1]),
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
class TestPrimitiveOp_b1c86ecb68e75819f3a9f95e02d40639(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 56, 80], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_7e116b205524dc783ac822a3e6326c3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.414215087890625]], [[3.9895942211151123]], [[3.3893890380859375]], [[3.991058349609375]], [[3.3325841426849365]], [[2.972386360168457]], [[4.130126953125]], [[3.263284921646118]], [[3.8213534355163574]], [[3.9842443466186523]], [[3.1546404361724854]], [[3.424934148788452]], [[3.3857316970825195]], [[3.1984293460845947]]]], dtype='float32').reshape([1, 14, 1, 1]),
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
class TestPrimitiveOp_aeafd0e0cfd32372fdd73d0c65940d86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_92f8b4133eb788c5ebcec984bc3b9df0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.825429439544678]], [[4.922501087188721]], [[4.990355968475342]], [[5.489058494567871]], [[4.323785305023193]], [[4.699490547180176]], [[5.001468658447266]], [[4.936253070831299]], [[4.6549482345581055]], [[4.829311370849609]], [[5.099071025848389]], [[4.964115142822266]], [[4.749936103820801]], [[4.554876804351807]], [[4.3140411376953125]], [[4.654892444610596]], [[5.267151355743408]], [[4.372352123260498]], [[5.095065593719482]], [[4.78563928604126]]]], dtype='float32').reshape([1, 20, 1, 1]),
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
class TestPrimitiveOp_55f7f318239192a5e698930498a7850c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_629a7fd69e5f24352af0c396a0870240(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[35703.07421875]], [[29847.548828125]], [[35764.2890625]], [[41210.7734375]], [[28388.34765625]], [[38335.7578125]]], [[[35184.9375]], [[29422.140625]], [[35248.55859375]], [[40622.1796875]], [[27982.34375]], [[37782.7890625]]]], dtype='float32').reshape([2, 6, 1, 1]),
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
class TestPrimitiveOp_fe85363c1ea28f9284ecdbeaf78c38c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[36232.578125]], [[32477.4140625]], [[38839.375]], [[36755.08984375]], [[48853.46875]], [[38602.6171875]]], [[[35125.6171875]], [[31485.537109375]], [[37655.41796875]], [[35633.1875]], [[47361.4296875]], [[37426.2265625]]]], dtype='float32').reshape([2, 6, 1, 1]),
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
class TestPrimitiveOp_10ad80d28b30d3b0d2d1d3bd79b33b79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[40847.02734375]], [[40062.1953125]], [[45187.515625]], [[47650.5703125]], [[41011.328125]], [[38968.39453125]]], [[[39265.29296875]], [[38511.7109375]], [[43442.58984375]], [[45809.91015625]], [[39421.35546875]], [[37462.34375]]]], dtype='float32').reshape([2, 6, 1, 1]),
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
class TestPrimitiveOp_c6a7a6339c8dbd4400d28ad8f6237a30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[37322.234375]], [[45104.25390625]], [[46323.05859375]], [[44561.17578125]], [[40449.51953125]], [[36808.828125]]], [[[36430.921875]], [[44031.66796875]], [[45224.703125]], [[43501.35546875]], [[39489.7109375]], [[35936.8515625]]]], dtype='float32').reshape([2, 6, 1, 1]),
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
class TestPrimitiveOp_0e05f9da72ceb520b50a48939f69b175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d4feb74016a940aadd744c0fb789d4c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_52693719c6145aa382af0595283d22c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_58a2d5338e7d43ffb926452e5fb47be5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_caf023224d64b5a9da3a7c071d58667c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.6658353805542]], [[8.1070556640625]], [[8.222827911376953]], [[7.877943992614746]], [[8.066951751708984]], [[7.292673110961914]], [[8.084744453430176]], [[7.5229811668396]], [[8.21318244934082]], [[7.604622840881348]], [[8.83447265625]], [[8.344555854797363]], [[9.45718002319336]], [[7.1796956062316895]], [[8.142525672912598]], [[7.744045734405518]], [[7.552488803863525]], [[7.6111650466918945]], [[7.037825107574463]], [[9.103200912475586]], [[8.932579040527344]], [[8.454848289489746]], [[7.616219997406006]], [[8.000482559204102]], [[7.608373165130615]], [[7.816816806793213]], [[8.14023494720459]], [[8.14525318145752]], [[7.936910629272461]], [[8.751760482788086]]]], dtype='float32').reshape([1, 30, 1, 1]),
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
class TestPrimitiveOp_d472d0fa1e160239a0512cec19704668(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.416786193847656]], [[7.794103622436523]], [[8.924464225769043]], [[7.228585243225098]], [[9.031545639038086]], [[8.148994445800781]], [[7.675243854522705]], [[7.80863618850708]], [[7.688236236572266]], [[8.47314453125]], [[7.7580342292785645]], [[7.796664714813232]], [[7.8048248291015625]], [[8.402878761291504]], [[7.834222316741943]], [[7.293983459472656]], [[7.622461795806885]], [[8.414867401123047]], [[8.357909202575684]], [[8.451399803161621]], [[6.94551944732666]], [[8.028322219848633]], [[7.767489910125732]], [[8.670419692993164]], [[7.389835357666016]], [[7.984164237976074]], [[8.085855484008789]], [[7.575388431549072]], [[7.201699733734131]], [[8.10418701171875]]]], dtype='float32').reshape([1, 30, 1, 1]),
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
class TestPrimitiveOp_5b587a6b5a79acfc31d910ed09b8cf67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 44, 66], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_df5256d41061334b86d740855120edc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.368825435638428]], [[7.740781307220459]], [[8.700937271118164]], [[8.02814769744873]], [[7.299408435821533]], [[7.574045181274414]], [[7.590500831604004]], [[7.792178630828857]], [[7.857649803161621]], [[7.405430793762207]], [[7.992948055267334]], [[7.77955436706543]], [[7.631563186645508]], [[7.997115135192871]], [[7.894710540771484]], [[7.477319240570068]], [[7.802051544189453]], [[7.672013282775879]], [[7.112525463104248]], [[7.324472904205322]], [[7.04571008682251]], [[8.032870292663574]], [[7.49959135055542]], [[8.095975875854492]], [[7.526414394378662]], [[8.05590534210205]], [[8.160518646240234]], [[8.33193302154541]], [[8.011821746826172]], [[7.693360328674316]]]], dtype='float32').reshape([1, 30, 1, 1]),
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
class TestPrimitiveOp_71d67eb974ef937e0f20713d0842d4b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_17c449821a6eb8ae41ad3959df65f543(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.7651166915893555]], [[7.80988883972168]], [[7.753073215484619]], [[7.550171852111816]], [[7.778337001800537]], [[7.513052463531494]], [[6.393861770629883]], [[8.28150749206543]], [[7.846116542816162]], [[7.819559097290039]], [[8.35317611694336]], [[8.397684097290039]], [[7.938183307647705]], [[7.887857437133789]], [[7.237251281738281]], [[7.872225284576416]], [[8.066829681396484]], [[8.189265251159668]], [[8.395303726196289]], [[8.470483779907227]], [[8.274739265441895]], [[8.872918128967285]], [[6.792125701904297]], [[7.824231147766113]], [[8.335457801818848]], [[7.598811626434326]], [[7.617952346801758]], [[8.056575775146484]], [[7.427498817443848]], [[8.475411415100098]]]], dtype='float32').reshape([1, 30, 1, 1]),
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
class TestPrimitiveOp_91c21af29ef3bca9cfb048ebd029c259(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.968533754348755]], [[3.2744359970092773]], [[4.22754430770874]], [[3.6164140701293945]], [[3.2631449699401855]], [[3.6471145153045654]], [[3.716449737548828]], [[3.3851146697998047]], [[3.310685634613037]], [[3.4721317291259766]], [[2.960721254348755]], [[3.1858975887298584]]]], dtype='float32').reshape([1, 12, 1, 1]),
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
class TestPrimitiveOp_6013a99e9ea120611a902d4a95f94f4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.6957502365112305]], [[2.9693329334259033]], [[2.8324947357177734]], [[3.0528671741485596]], [[2.9085195064544678]], [[3.041714668273926]], [[2.9788265228271484]], [[3.414797306060791]], [[3.3351552486419678]], [[3.358827590942383]], [[3.3882534503936768]], [[3.147059440612793]]]], dtype='float32').reshape([1, 12, 1, 1]),
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
class TestPrimitiveOp_21ef9f445fac83a3be92d924a60b01d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.293083190917969]], [[5.870551109313965]], [[6.320681571960449]], [[6.907689571380615]], [[6.797855377197266]], [[6.495174407958984]], [[5.911848068237305]], [[6.560781955718994]], [[6.150582313537598]], [[6.502373218536377]], [[6.399684429168701]], [[5.962441921234131]], [[6.735020637512207]], [[7.271669387817383]], [[6.681186199188232]], [[6.115781307220459]], [[6.6929850578308105]], [[7.5167646408081055]], [[6.415417671203613]], [[6.338051795959473]], [[6.1207966804504395]], [[6.665828704833984]], [[6.196653366088867]], [[6.108193397521973]], [[7.369046211242676]]]], dtype='float32').reshape([1, 25, 1, 1]),
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
class TestPrimitiveOp_b47adf61accad47610565706c0712cbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_eca32c4a11018f1951438540b4e8fdb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([1, 312], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f1b45f67fe34021a70d6a79a51075505(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([171, 120], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_3b5b229243a9e9f3b345f7858600d2cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([145, 9], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8bbf899625f58e211ff73edb68db3f43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 5, 8], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1ea63d1341730c5351917b71b6dd0d11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.721285343170166]], [[4.808663368225098]], [[3.7746777534484863]], [[4.069631576538086]], [[3.8913345336914062]], [[4.651084899902344]], [[4.408143520355225]], [[3.8639371395111084]], [[4.055018424987793]], [[3.73452091217041]], [[4.03886604309082]], [[3.93035626411438]], [[4.237965106964111]], [[4.525489330291748]], [[4.1659650802612305]], [[3.751066207885742]], [[3.5675864219665527]], [[3.557084083557129]]]], dtype='float32').reshape([1, 18, 1, 1]),
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
class TestPrimitiveOp_4e3e5bd3b7aa5ec9b9d4430d8378b714(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([1, 39], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6edefde6ef978e8ad10b2cd468805dd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.8998667001724243]], [[1.3319482803344727]], [[1.4812557697296143]], [[1.1986289024353027]], [[1.647526502609253]]]], dtype='float32').reshape([1, 5, 1, 1]),
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
class TestPrimitiveOp_374fe640996bf8c89b12a6dc0b48da0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.685392141342163]], [[3.432683229446411]], [[2.8449554443359375]], [[2.8167319297790527]], [[2.9428277015686035]], [[2.992644786834717]], [[2.637897491455078]], [[2.3180813789367676]], [[2.680163860321045]], [[3.168722629547119]]]], dtype='float32').reshape([1, 10, 1, 1]),
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
class TestPrimitiveOp_9b91ceac60dcb0d2c3ece364b8cecb33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.834033489227295]], [[6.028222560882568]], [[6.2730607986450195]], [[5.815683364868164]], [[5.537166595458984]], [[5.755615234375]], [[5.125760078430176]], [[4.9682207107543945]], [[5.588512897491455]], [[6.035160064697266]], [[4.70105504989624]], [[5.620141506195068]], [[5.955061912536621]], [[5.09976863861084]], [[5.872868537902832]], [[6.469480991363525]], [[5.581788539886475]], [[5.994884967803955]], [[5.6630120277404785]], [[5.77250337600708]]]], dtype='float32').reshape([1, 20, 1, 1]),
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
class TestPrimitiveOp_78e0f65d3cbe51919e1385a22e23c598(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.5755462646484375]], [[7.724958419799805]], [[6.630558490753174]], [[7.034902572631836]], [[6.702817916870117]], [[6.833985328674316]], [[6.586999416351318]], [[6.8699631690979]], [[6.244996547698975]], [[6.929019451141357]], [[7.096578598022461]], [[6.641002178192139]], [[6.750032901763916]], [[6.835293769836426]], [[6.002835273742676]], [[7.5838422775268555]], [[7.019974708557129]], [[6.836799621582031]], [[6.463237285614014]], [[7.26837682723999]], [[6.874286651611328]], [[7.179231643676758]], [[6.164690017700195]], [[7.2050933837890625]]]], dtype='float32').reshape([1, 24, 1, 1]),
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
class TestPrimitiveOp_2041013682eaded7ecacb301e2e58a3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([22, 120], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_0fd86d1a7e48dba6d098dc059fffa4ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.720344066619873]], [[2.8081295490264893]], [[2.952676296234131]], [[2.218912124633789]], [[2.8770737648010254]], [[2.590301990509033]], [[2.5677194595336914]], [[2.233036518096924]], [[2.099064350128174]], [[2.477353096008301]]]], dtype='float32').reshape([1, 10, 1, 1]),
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
class TestPrimitiveOp_15d54eb686939dd6ee056bac3ec425ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([145, 120], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_3ab070c6f1ad19b24d4dab303ab2db11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 40, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_207ced3fb8aef9175b31f81beb2a8256(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 50, 76], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_07d2a7b2504ddc0eacb4766e4ce98dbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([171, 9], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d691ea85ddcd090d3ecce8e12e1e1903(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.405776023864746]], [[4.127753257751465]], [[4.362503528594971]], [[4.529914855957031]], [[4.277790069580078]], [[4.447083473205566]], [[4.63477897644043]], [[4.138066291809082]], [[3.7077553272247314]], [[5.377406120300293]], [[3.928138494491577]], [[4.060510635375977]], [[4.17971134185791]], [[4.1579790115356445]], [[3.9366140365600586]], [[4.897757053375244]], [[4.514909267425537]], [[4.128507137298584]]]], dtype='float32').reshape([1, 18, 1, 1]),
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
class TestPrimitiveOp_5146fd90b333ffb903c2a1c80caa8194(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.to_tensor([[8.378665924072266, 8.622936248779297, 7.992419242858887, 7.285286903381348, 7.504433631896973, 7.695581912994385, 7.320318698883057, 7.855299949645996, 7.528284072875977, 7.803767681121826, 8.484917640686035, 8.615951538085938, 7.278975009918213, 7.6329498291015625, 7.102418422698975, 7.5837626457214355, 7.093390941619873, 8.706185340881348, 8.39137077331543, 7.464969158172607, 8.560622215270996, 8.868217468261719, 7.718655586242676, 8.08227825164795, 7.667810440063477, 7.491669178009033, 7.475050449371338, 7.9493088722229, 8.244874000549316, 7.5856757164001465]], dtype='float32').reshape([1, 30]),
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
class TestPrimitiveOp_48b5fd3b41954aaf9d3135972ac3942e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.708158016204834]], [[7.252604961395264]], [[7.377367973327637]], [[6.789035320281982]], [[7.606326580047607]], [[7.113602638244629]], [[6.4286789894104]], [[6.917911529541016]], [[7.2128777503967285]], [[7.766190528869629]], [[6.791611194610596]], [[6.892510890960693]], [[6.806152820587158]], [[7.006817817687988]], [[7.4687395095825195]], [[5.781470775604248]], [[6.6284332275390625]], [[7.578555107116699]], [[6.092510223388672]], [[7.131056308746338]], [[5.506556034088135]], [[6.615859031677246]], [[6.5423197746276855]], [[6.561517238616943]], [[6.8081374168396]], [[6.281248092651367]], [[7.557594299316406]], [[6.767190456390381]], [[7.261711120605469]], [[6.651391983032227]]]], dtype='float32').reshape([1, 30, 1, 1]),
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
class TestPrimitiveOp_b6d66fdd2cde251e1bfab140c1985ed3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.7736645936965942]], [[1.3818273544311523]], [[1.6397769451141357]], [[1.2828425168991089]], [[1.889788031578064]]]], dtype='float32').reshape([1, 5, 1, 1]),
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
class TestPrimitiveOp_65dbd0a098823e9d6ef1419dfefb1814(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.642515182495117]], [[2.7344796657562256]], [[2.4162209033966064]], [[3.6685612201690674]], [[2.6550488471984863]], [[2.7155582904815674]], [[2.9602038860321045]], [[3.0387861728668213]], [[3.186140775680542]], [[3.3328189849853516]]]], dtype='float32').reshape([1, 10, 1, 1]),
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
class TestPrimitiveOp_930533a9539057ffd632ed674b43e601(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.030589580535889]], [[5.035013675689697]], [[4.420341491699219]], [[5.301993370056152]], [[5.071216106414795]], [[5.969995021820068]], [[5.171087265014648]], [[6.536515235900879]], [[5.798988342285156]], [[5.952205181121826]], [[6.107416152954102]], [[6.40071964263916]], [[5.373867034912109]], [[5.728961944580078]], [[5.953717231750488]], [[5.541092395782471]], [[5.195215225219727]], [[6.765068054199219]], [[5.535505771636963]], [[4.950847148895264]]]], dtype='float32').reshape([1, 20, 1, 1]),
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
class TestPrimitiveOp_5e50ceb17c54cdd8788aa1f2f7a54da8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.375254154205322]], [[3.876783609390259]], [[4.515564441680908]], [[4.219099998474121]], [[4.444534778594971]], [[3.815962553024292]], [[4.503503322601318]], [[4.283707618713379]], [[4.291059494018555]], [[4.729404926300049]], [[4.597497940063477]], [[4.576755046844482]], [[4.495851039886475]], [[4.035416603088379]], [[3.8874900341033936]], [[4.451059341430664]]]], dtype='float32').reshape([1, 16, 1, 1]),
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
class TestPrimitiveOp_05bbbedf805471186bd46f14cb402376(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c1c8322b540a576992822cdf8c9bbec4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b9dfcf9e6ae0639a3afa602e2f7fc88c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_07338898e9a2394aa61e1cdcf2f8a24f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5e3d0b7da51a15124d9cb453b329ce20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.194644927978516]], [[3.3138089179992676]], [[4.333257675170898]], [[3.9553563594818115]], [[4.612022399902344]], [[4.258015155792236]], [[4.4418416023254395]], [[4.1807708740234375]], [[4.294811248779297]], [[4.170912742614746]], [[4.017587184906006]], [[4.550390243530273]], [[3.9489927291870117]], [[3.9343838691711426]]]], dtype='float32').reshape([1, 14, 1, 1]),
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
class TestPrimitiveOp_5d2269fef135f4c8d760dbbc6414b4e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.218982219696045]], [[5.918820858001709]], [[5.5988874435424805]], [[4.726998329162598]], [[5.55240535736084]], [[6.374054431915283]], [[5.321900844573975]], [[5.318398475646973]], [[5.078855991363525]], [[5.079309463500977]], [[5.932351112365723]], [[6.2274298667907715]], [[5.890310764312744]], [[5.522193431854248]], [[5.292682647705078]], [[5.201898574829102]], [[5.938455581665039]], [[5.890253067016602]], [[4.985781192779541]], [[5.914712429046631]]]], dtype='float32').reshape([1, 20, 1, 1]),
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
class TestPrimitiveOp_cbaa91c541da991cc1283089be6abbcc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 14, 20], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ea99efb57e9e2c1124ee1286bc0d989c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.02066421508789]], [[7.951235294342041]], [[7.977724075317383]], [[8.337994575500488]], [[8.380784034729004]], [[9.448760032653809]], [[8.62950325012207]], [[8.02839183807373]], [[8.133667945861816]], [[7.411727428436279]], [[7.980532646179199]], [[8.305061340332031]], [[8.481549263000488]], [[7.9515485763549805]], [[7.594925403594971]], [[7.8966498374938965]], [[8.162572860717773]], [[7.834351062774658]], [[7.796360969543457]], [[7.60296106338501]], [[8.017879486083984]], [[7.912840366363525]], [[7.13515567779541]], [[7.504832744598389]], [[8.09492015838623]], [[7.242308616638184]], [[8.097259521484375]], [[7.662278652191162]], [[8.42585563659668]], [[7.133913993835449]]]], dtype='float32').reshape([1, 30, 1, 1]),
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
class TestPrimitiveOp_90b2665146e8acddcc311ab2513215e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([22, 96, 109, 109], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5b25216e947d95e442ae7a675d642f37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 54, 54], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6b3b6953c815bb0efd817664f0381585(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_77bffecbd6c38445f94e51244b00e5a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 54, 54], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_93c3fd2bd4824ccd9c65719d06eea1c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 54, 54], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d080da28eebcf475c3447482f55d8f97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_327aeff3e39beb3bbe908f7425094dab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_99343f3221b77313058352e27c6eeeb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4f5f676f44960e31eacd927755dec29b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_df27faebd82272f6979f0cf6d0d6990b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_fe78f8fb8d0628309513a3b18f0a4449(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ba4f9ca46f6928483d4fe04b5f969974(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_02eb531dd27598c3cb929e534053d91b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_117a6965472cf9cdf2d8ee92fcc11842(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_7037d3650459f14dbe4accfeabe669e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 16], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_21f2b3e8b267836c105e9a20a0b40818(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.767958641052246]], [[6.259011268615723]], [[6.5256195068359375]], [[5.72200345993042]], [[6.045093536376953]], [[6.097092628479004]], [[6.170426368713379]], [[5.24308967590332]], [[5.914750099182129]], [[6.250386714935303]], [[6.673716068267822]], [[6.721867084503174]], [[5.911425590515137]], [[6.5793328285217285]], [[6.046773433685303]], [[6.206075191497803]], [[6.1781110763549805]], [[7.2541093826293945]], [[5.930622577667236]], [[5.391793727874756]], [[5.651362895965576]], [[6.339712619781494]], [[5.226492404937744]], [[5.5235443115234375]]]], dtype='float32').reshape([1, 24, 1, 1]),
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
class TestPrimitiveOp_b7a55c86141dfc05a8b5b98798fd0dd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.254484176635742]], [[6.779487609863281]], [[5.887111186981201]], [[6.078207492828369]], [[7.187652111053467]], [[6.1240434646606445]], [[6.604331016540527]], [[7.094372749328613]], [[6.038271903991699]], [[6.957431793212891]], [[6.0269670486450195]], [[7.332108497619629]], [[6.249767780303955]], [[6.39636754989624]], [[7.155977249145508]], [[6.703110218048096]], [[5.996480941772461]], [[7.105651378631592]], [[5.985050201416016]], [[5.435423851013184]], [[6.244361400604248]], [[5.577609539031982]], [[5.8099589347839355]], [[5.968587875366211]], [[6.123788833618164]]]], dtype='float32').reshape([1, 25, 1, 1]),
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
class TestPrimitiveOp_afd39ae4a2034fa8723c70ab6f61bacb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.888890504837036]], [[3.3929007053375244]], [[2.6112351417541504]], [[3.3341012001037598]], [[2.9060111045837402]], [[3.714590072631836]], [[3.5299689769744873]], [[2.686753511428833]], [[2.849156379699707]], [[3.0124778747558594]], [[2.9774067401885986]], [[3.0496160984039307]]]], dtype='float32').reshape([1, 12, 1, 1]),
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
class TestPrimitiveOp_ce8267680df7a42e6226644a4b6a24da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 25, 38], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_00b3e905cc899e8521f5b9bd640efed2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 112, 160], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5d106b0689815e7f536b6c24a0dfff88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 7, 10], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b0110fdd0aafae9e8bce4be4ee8942d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[655.7572021484375]], [[705.666015625]], [[773.6893920898438]], [[709.565673828125]], [[721.9839477539062]], [[814.34814453125]], [[689.0104370117188]], [[783.5064697265625]], [[732.5365600585938]], [[706.2600708007812]], [[773.250244140625]], [[794.0547485351562]], [[746.3873901367188]], [[696.74853515625]], [[749.4446411132812]], [[793.5221557617188]], [[676.5300903320312]], [[676.3561401367188]], [[726.6395263671875]], [[715.7932739257812]], [[653.9281005859375]], [[629.6777954101562]], [[765.410400390625]], [[767.4696655273438]]]], dtype='float32').reshape([1, 24, 1, 1]),
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
class TestPrimitiveOp_5e8848bc614cc40d9ad24f9f20e08e0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[74.81359100341797]], [[78.00292205810547]], [[86.67084503173828]], [[90.57637786865234]], [[75.38327026367188]], [[83.65338897705078]], [[88.43961334228516]], [[84.3248062133789]], [[84.56562042236328]], [[86.31048583984375]], [[82.31513977050781]], [[77.43717193603516]], [[79.58155059814453]], [[86.9212417602539]], [[87.23821258544922]], [[82.0116195678711]], [[83.5269775390625]], [[92.50822448730469]], [[90.42761993408203]], [[90.77925872802734]], [[82.18746185302734]], [[81.39210510253906]], [[83.22457885742188]], [[90.01710510253906]]]], dtype='float32').reshape([1, 24, 1, 1]),
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
class TestPrimitiveOp_3c73d306b7b332b8d4945adc2a185db1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[33.23279571533203]], [[30.137006759643555]], [[29.386131286621094]], [[25.653465270996094]], [[32.61014938354492]], [[26.98526954650879]], [[31.62348175048828]], [[29.61621856689453]], [[31.396163940429688]], [[32.07232666015625]], [[29.105955123901367]], [[30.439414978027344]], [[30.328889846801758]], [[28.05185317993164]], [[31.804729461669922]], [[28.109773635864258]], [[31.42383575439453]], [[29.110925674438477]], [[28.396617889404297]], [[27.790151596069336]], [[26.741432189941406]], [[30.17379379272461]], [[29.696020126342773]], [[33.10212326049805]]]], dtype='float32').reshape([1, 24, 1, 1]),
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
class TestPrimitiveOp_b527c1bfff3ac71511dc5a78fb38b877(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[31.59615707397461]], [[32.04358673095703]], [[29.37789535522461]], [[31.304113388061523]], [[31.302127838134766]], [[29.74681282043457]], [[32.08837890625]], [[31.049131393432617]], [[31.736263275146484]], [[29.832345962524414]], [[32.678524017333984]], [[28.981279373168945]], [[30.287275314331055]], [[29.703840255737305]], [[29.753652572631836]], [[28.28385353088379]], [[30.456066131591797]], [[33.913978576660156]], [[31.88187599182129]], [[30.4820556640625]], [[28.127578735351562]], [[30.154258728027344]], [[31.139921188354492]], [[29.98908233642578]]]], dtype='float32').reshape([1, 24, 1, 1]),
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
class TestPrimitiveOp_ed2800fa24b378c25a78359c528013db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[32348.275390625]], [[35888.19921875]], [[37447.94921875]], [[28942.8515625]], [[34149.21484375]], [[28461.111328125]]]], dtype='float32').reshape([1, 6, 1, 1]),
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
class TestPrimitiveOp_6e1733c6e071ed09cf5b17b56571deb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[39232.3828125]], [[35451.703125]], [[41437.140625]], [[39485.75]], [[39789.12109375]], [[44328.6796875]]]], dtype='float32').reshape([1, 6, 1, 1]),
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
class TestPrimitiveOp_53a2ba6580a1edeae3429cc26d3a7a57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[48054.35546875]], [[46217.55859375]], [[37226.0390625]], [[40559.53125]], [[43859.953125]], [[41513.4921875]]]], dtype='float32').reshape([1, 6, 1, 1]),
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
class TestPrimitiveOp_ad182a55a08a6decf0df6037a17d58dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[48646.22265625]], [[41030.73046875]], [[43503.50390625]], [[41555.27734375]], [[38020.99609375]], [[36831.12109375]]]], dtype='float32').reshape([1, 6, 1, 1]),
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
class TestPrimitiveOp_ba0c4f481c77bb3567197468e4202dd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 11, 17], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b6cd80f75f7eafb0ae48846efa065a85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 88, 132], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_e28f5d4cc05484515e227487930c799e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.077600955963135]], [[6.519475936889648]], [[6.887722015380859]], [[6.376555442810059]], [[6.385010242462158]], [[7.068292140960693]], [[7.098506927490234]], [[5.753015518188477]], [[6.571295738220215]], [[7.231013298034668]], [[7.227197647094727]], [[6.2340474128723145]], [[6.135470867156982]], [[7.156917095184326]], [[6.59360933303833]], [[6.4501142501831055]], [[6.219383716583252]], [[6.524694442749023]], [[7.338897705078125]], [[6.9839887619018555]], [[6.738530158996582]], [[6.914845943450928]], [[7.0328049659729]], [[6.626660346984863]]]], dtype='float32').reshape([1, 24, 1, 1]),
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
class TestPrimitiveOp_59862ab7f1c98eb3ee597018396bad7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f0ecf685d0887f6a5083a8c4788b902
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 100, 152], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_14cd5d8161c3c4181f532187e10dd36a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22381fbd1daa84325ea8679549f1bef6
    def get_inputs(self):
        return [
            paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_4ddcce8a180b8c33a60631653546bc50(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d3f444a16e823bb7d0f88307f2dc0a06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ddcce8a180b8c33a60631653546bc50
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_4c693a4a2ff78966743a88daf201a080(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 18], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ffdeb6e59c7d8bb399da624577323248(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c693a4a2ff78966743a88daf201a080
    def get_inputs(self):
        return [
            paddle.to_tensor([[4.685884952545166, 4.341721534729004, 5.0854668617248535, 4.807162284851074, 4.381277561187744, 4.854126453399658, 4.59234094619751, 4.803297519683838, 4.440852642059326, 4.615706443786621, 3.8427395820617676, 5.17818546295166, 4.46326208114624, 4.0751471519470215, 4.623128890991211, 4.63656759262085, 5.268134117126465, 4.743747711181641]], dtype='float32').reshape([1, 18]),
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

class PrimitiveOp_c33ce085e6bf5d9ed0a32609edf6f9dc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 23], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9d874a71609f91b5ca8ef323780b4b50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c33ce085e6bf5d9ed0a32609edf6f9dc
    def get_inputs(self):
        return [
            paddle.to_tensor([[6.394744873046875, 5.274114608764648, 5.703514099121094, 6.524235248565674, 5.836944103240967, 6.776856422424316, 6.140979766845703, 6.095730781555176, 5.962957859039307, 6.768327236175537, 6.455108642578125, 6.547737121582031, 5.934031009674072, 5.640436172485352, 6.510316371917725, 5.865507125854492, 5.536918640136719, 5.9943742752075195, 5.177901744842529, 6.1067633628845215, 6.230833530426025, 5.81798791885376, 6.211367130279541]], dtype='float32').reshape([1, 23]),
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

class PrimitiveOp_c5eef8ac6424fc66bad7653eaeda28b2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 40, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_61d8c5843ee71a4304fc62712513d2c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5eef8ac6424fc66bad7653eaeda28b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_d281a4a803bf8d92f7b832b52b52b462(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_df92cfef751e75c1c138a97f2a850274(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d281a4a803bf8d92f7b832b52b52b462
    def get_inputs(self):
        return [
            paddle.uniform([1, 240], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_4d787b651e173fdd72b62c54eb2369a9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 120], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ce77de927db7aa386add706b4800d5a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d787b651e173fdd72b62c54eb2369a9
    def get_inputs(self):
        return [
            paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_97a676267ac0217e9a726cdb750b02d6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_abfa5feb67cf192fe83d2f6419a68987(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97a676267ac0217e9a726cdb750b02d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 20, 32], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_a6e0ff18eb999e2b18c1238c9039b985(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_09cde8ab57639fa9db5b1f333ea0bf05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6e0ff18eb999e2b18c1238c9039b985
    def get_inputs(self):
        return [
            paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_46c6c96cf66b0c7a3d16150e304f6ce0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 168, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_182d33db9ba31b1536ed2b4b293b153d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46c6c96cf66b0c7a3d16150e304f6ce0
    def get_inputs(self):
        return [
            paddle.uniform([1, 168, 1, 1], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_54278e400b2be87ca05c602cb1ca8126(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 30, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ca349dffbe0d366433d51a2894aff0e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54278e400b2be87ca05c602cb1ca8126
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.709053039550781]], [[7.550483703613281]], [[7.344411849975586]], [[6.536085605621338]], [[7.039064407348633]], [[6.679584503173828]], [[7.219615936279297]], [[7.443569660186768]], [[7.062199115753174]], [[6.990291118621826]], [[7.615472793579102]], [[6.663925647735596]], [[6.661057949066162]], [[7.0189337730407715]], [[7.696951866149902]], [[7.372894763946533]], [[7.589016914367676]], [[7.98672342300415]], [[7.5524468421936035]], [[7.484039783477783]], [[7.152288913726807]], [[7.274065971374512]], [[7.17875862121582]], [[6.769964218139648]], [[7.289031028747559]], [[7.488609313964844]], [[7.0675153732299805]], [[7.440677642822266]], [[7.4233222007751465]], [[6.610384464263916]]]], dtype='float32').reshape([1, 30, 1, 1]),
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

class PrimitiveOp_262ba2151df345eb3c5ae0bd348a8026(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 84], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eb534e011f7fd193685926449ddab4ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262ba2151df345eb3c5ae0bd348a8026
    def get_inputs(self):
        return [
            paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_a324f85e300e87f041a35ba8fa34127e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_66f128c8010ef1b87790aed3ae2520fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_fd6df46a1cfba23274339f5962d9926a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8dbf55bf23098fcbb395faafe55f697c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a46b4f0d5d188955d59c82278df6b70f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 15], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_74d0080cabdd20368c3e36d951dcf3d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 8], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9a813e9c1a8f5fd960c968a77c7a02c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54278e400b2be87ca05c602cb1ca8126
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.439476013183594]], [[7.116119861602783]], [[7.933470726013184]], [[7.429988384246826]], [[7.739742755889893]], [[7.160177230834961]], [[6.873104572296143]], [[6.83212423324585]], [[8.443646430969238]], [[7.169576644897461]], [[7.918055534362793]], [[8.18014144897461]], [[7.301506519317627]], [[7.215599060058594]], [[7.0682148933410645]], [[7.000136375427246]], [[8.593505859375]], [[6.998315334320068]], [[7.606921672821045]], [[7.383068084716797]], [[7.5837273597717285]], [[7.888897895812988]], [[7.603029251098633]], [[8.677313804626465]], [[8.388718605041504]], [[7.459686756134033]], [[8.030505180358887]], [[7.010253429412842]], [[8.303638458251953]], [[7.043381214141846]]]], dtype='float32').reshape([1, 30, 1, 1]),
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
class TestPrimitiveOp_3e10bfd0ec360e2d21613f0c79a8d420(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97a676267ac0217e9a726cdb750b02d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 50, 76], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_ccabd4bdc8c15e1883d6b58f2ffcf4ee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 5, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_475d63e38cea1b1dd02c8ff788d4cdf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccabd4bdc8c15e1883d6b58f2ffcf4ee
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.5144989490509033]], [[1.4872188568115234]], [[1.3116577863693237]], [[1.1540439128875732]], [[1.335241675376892]]]], dtype='float32').reshape([1, 5, 1, 1]),
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

class PrimitiveOp_db3a420cbd0dbb845b550b1a09e56109(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 10, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_668d9fa3ed9b2ed011673f092fd04a07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db3a420cbd0dbb845b550b1a09e56109
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.831326484680176]], [[2.8996922969818115]], [[2.341322422027588]], [[2.725881338119507]], [[2.1794872283935547]], [[2.3146777153015137]], [[3.153815269470215]], [[2.625821352005005]], [[2.693358898162842]], [[3.4684395790100098]]]], dtype='float32').reshape([1, 10, 1, 1]),
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

class PrimitiveOp_889d3db9c1718c4c87c2966a10bba689(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_724d888912d9f3f997ccf2be87c6634c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_889d3db9c1718c4c87c2966a10bba689
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_80ed7330d369c1f34d273814928826ae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 24, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9f5854374a52a65b7d888d40ff89fdbb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80ed7330d369c1f34d273814928826ae
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.052102088928223]], [[5.973732948303223]], [[6.436169624328613]], [[5.944063663482666]], [[6.007678508758545]], [[6.059599876403809]], [[5.8827714920043945]], [[6.284278869628906]], [[6.138669013977051]], [[5.883573055267334]], [[5.338260173797607]], [[5.5233683586120605]], [[5.554606914520264]], [[5.810969352722168]], [[5.240361213684082]], [[6.271805286407471]], [[6.381803512573242]], [[6.284727096557617]], [[6.23016357421875]], [[5.052532196044922]], [[5.897260665893555]], [[5.679372310638428]], [[5.945131301879883]], [[6.913031101226807]]]], dtype='float32').reshape([1, 24, 1, 1]),
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
class TestPrimitiveOp_7ac1d6bd59f07b7fb39f37ae57f4aadd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97a676267ac0217e9a726cdb750b02d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 100, 152], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_600a81d6fcfd3442889b3d9653a28b6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 13, 19], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_bbde9be5291cab7a77e92af8048f17ec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 15], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ad0afab41ade050dd03dc2e839091b94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbde9be5291cab7a77e92af8048f17ec
    def get_inputs(self):
        return [
            paddle.uniform([10, 15], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_bf5d9f0bba7a7b73e7e38ae50c877229(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 18, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3e9b058d53fa24da69c4a2fe5dc08c58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf5d9f0bba7a7b73e7e38ae50c877229
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.624106407165527]], [[4.701679706573486]], [[3.785703659057617]], [[4.636858940124512]], [[4.185674667358398]], [[5.190023422241211]], [[3.895230293273926]], [[3.9510366916656494]], [[4.272900581359863]], [[4.303114891052246]], [[4.5292229652404785]], [[3.799717664718628]], [[4.855134963989258]], [[4.441648006439209]], [[4.650888442993164]], [[4.795456886291504]], [[4.847243309020996]], [[4.663857460021973]]]], dtype='float32').reshape([1, 18, 1, 1]),
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
class TestPrimitiveOp_b6625c6776e0abb3abae884e83eed394(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80ed7330d369c1f34d273814928826ae
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.78853702545166]], [[7.146003246307373]], [[6.132681846618652]], [[6.221840858459473]], [[6.431774616241455]], [[6.59204626083374]], [[6.623891353607178]], [[6.397099494934082]], [[6.569820880889893]], [[7.090774059295654]], [[5.756249904632568]], [[5.597261905670166]], [[5.9224958419799805]], [[5.9571452140808105]], [[5.673080921173096]], [[6.735020637512207]], [[6.120677947998047]], [[6.651705741882324]], [[6.907902717590332]], [[6.890196800231934]], [[6.471126079559326]], [[6.618170261383057]], [[5.1616950035095215]], [[7.4771199226379395]]]], dtype='float32').reshape([1, 24, 1, 1]),
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
class TestPrimitiveOp_0ae9794f8234e3548ee8dd15adbc120b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262ba2151df345eb3c5ae0bd348a8026
    def get_inputs(self):
        return [
            paddle.uniform([145, 84], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_afb1a0e70110e434b78afd9f65a710fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97a676267ac0217e9a726cdb750b02d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 28, 40], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_474a628119c59d06cc8fb2eb2a66ddb2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_efbe8ebcfc91f2348598685d96ace5e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_474a628119c59d06cc8fb2eb2a66ddb2
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.544837236404419]], [[0.8169614672660828]], [[1.2499406337738037]], [[1.3618935346603394]]]], dtype='float32').reshape([1, 4, 1, 1]),
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

class PrimitiveOp_d0a332f3f65a0baa0b24c0fb0a72a04c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 11, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7df9936ab418d9671847bea54cff4505(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d0a332f3f65a0baa0b24c0fb0a72a04c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.406261444091797]], [[2.7747883796691895]], [[3.009711980819702]], [[3.176870346069336]], [[2.6877174377441406]], [[3.1237170696258545]], [[2.2043473720550537]], [[2.562354564666748]], [[3.4220175743103027]], [[3.2416279315948486]], [[2.9616148471832275]]]], dtype='float32').reshape([1, 11, 1, 1]),
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

class PrimitiveOp_19359c54ba9fa814761aadfec38e9057(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 120, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9e5829195b5969957bbc2b8cd268588e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19359c54ba9fa814761aadfec38e9057
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_44c7e78bf2353ed5f4e7a0173f5229e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54278e400b2be87ca05c602cb1ca8126
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.471386909484863]], [[7.349374771118164]], [[7.80717134475708]], [[6.732783794403076]], [[7.302953720092773]], [[7.446574687957764]], [[8.105687141418457]], [[8.2003812789917]], [[7.494767189025879]], [[7.934525966644287]], [[7.961569786071777]], [[6.840730667114258]], [[7.564905166625977]], [[7.182055473327637]], [[7.827128887176514]], [[7.650530815124512]], [[7.258990287780762]], [[7.523951053619385]], [[7.0820393562316895]], [[6.465030193328857]], [[6.736588478088379]], [[7.990978240966797]], [[7.304441452026367]], [[7.74916934967041]], [[6.747164249420166]], [[8.26745891571045]], [[7.089442729949951]], [[7.226877212524414]], [[7.833673000335693]], [[7.621578693389893]]]], dtype='float32').reshape([1, 30, 1, 1]),
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

class PrimitiveOp_015f60f0f7bb453319399c250296ff51(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 60], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cbc5d5625170ea7c723848b0db9a62b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_015f60f0f7bb453319399c250296ff51
    def get_inputs(self):
        return [
            paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_72466d6b0aac72c20d4bea6f95be44f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97a676267ac0217e9a726cdb750b02d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 80, 128], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_0a30ddc94d09c5f2cfeb153b79834946(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0ba48670ca3af304318f49db0e32bce5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a30ddc94d09c5f2cfeb153b79834946
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.8788163661956787]], [[4.8822431564331055]], [[4.714454174041748]], [[5.47055721282959]], [[5.005043029785156]], [[4.7693023681640625]], [[4.298551559448242]], [[4.079020023345947]], [[4.330775737762451]], [[4.043824195861816]], [[4.479321479797363]], [[4.7139410972595215]], [[4.095968246459961]], [[5.0010576248168945]], [[4.4950785636901855]], [[4.312562942504883]]]], dtype='float32').reshape([1, 16, 1, 1]),
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
class TestPrimitiveOp_21477458ce0127db95f5c638beed616f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97a676267ac0217e9a726cdb750b02d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 14, 20], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_00c56c84749fe84060ac6ef515bb1b5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97a676267ac0217e9a726cdb750b02d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 22, 33], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_24bd208a8b30167aa14a93a6b5110e16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97a676267ac0217e9a726cdb750b02d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 23, 35], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_0b3f6835f9fb139652b66a41978e8bb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97a676267ac0217e9a726cdb750b02d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 46, 70], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_2869441ec7d1d254bf3b5a013fe440c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbde9be5291cab7a77e92af8048f17ec
    def get_inputs(self):
        return [
            paddle.uniform([22, 15], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_c513ae7f47e626fe4801c31de815b10d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_558d75a863a543e1d813e0e31cebba1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c513ae7f47e626fe4801c31de815b10d
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_aee949c9ad97fe4f33777f356f38f15a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54278e400b2be87ca05c602cb1ca8126
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.351052284240723]], [[9.076446533203125]], [[7.611941337585449]], [[9.164783477783203]], [[7.442530155181885]], [[8.784290313720703]], [[8.335529327392578]], [[8.586498260498047]], [[8.109635353088379]], [[7.509426116943359]], [[8.61800479888916]], [[8.74172306060791]], [[7.5675435066223145]], [[8.286055564880371]], [[8.676008224487305]], [[8.710409164428711]], [[8.41573715209961]], [[8.553010940551758]], [[9.207554817199707]], [[8.445959091186523]], [[8.4314603805542]], [[9.244460105895996]], [[8.93134593963623]], [[7.5370073318481445]], [[8.284849166870117]], [[8.928435325622559]], [[9.228302001953125]], [[7.232161998748779]], [[8.493978500366211]], [[8.002117156982422]]]], dtype='float32').reshape([1, 30, 1, 1]),
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

class PrimitiveOp_7e04c7dfa61283a44e2d457562974ff9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 80, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2b777266b2716abf22b892a797401db1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e04c7dfa61283a44e2d457562974ff9
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_2c026065d8c14e23899bdb4a05c680b5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 218], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0e79a8ac8e1db56f935e6f400640b05b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c026065d8c14e23899bdb4a05c680b5
    def get_inputs(self):
        return [
            paddle.uniform([1, 218], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_2ad8d2f145fdbb6389a6e940b3876077(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 25, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_174a77f8f9ea23a24e625bc318755efe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ad8d2f145fdbb6389a6e940b3876077
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.523366451263428]], [[6.616939544677734]], [[6.126245021820068]], [[5.82805061340332]], [[6.883699417114258]], [[7.409236907958984]], [[6.320303916931152]], [[6.522290229797363]], [[6.37861442565918]], [[6.731780529022217]], [[6.4535746574401855]], [[6.217649936676025]], [[5.953060150146484]], [[5.921178340911865]], [[7.176547050476074]], [[7.22666072845459]], [[7.4815850257873535]], [[6.360232353210449]], [[6.470932483673096]], [[6.494004726409912]], [[6.564431190490723]], [[6.752301216125488]], [[6.501801013946533]], [[6.448027610778809]], [[6.599648475646973]]]], dtype='float32').reshape([1, 25, 1, 1]),
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
class TestPrimitiveOp_4fdab81655a8e914d80e4f21b1d33cae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97a676267ac0217e9a726cdb750b02d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 6, 9], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_e35d8561d7b9961ef31d2921a8e43f41(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5c13c7c46462356ba91d80379403c783(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e35d8561d7b9961ef31d2921a8e43f41
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ad7eda4a1aa1e7a6d8c599c1d597cc3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6e0ff18eb999e2b18c1238c9039b985
    def get_inputs(self):
        return [
            paddle.uniform([390, 1024], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_fdcf158a51c1d35473da1e39b28f6445(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262ba2151df345eb3c5ae0bd348a8026
    def get_inputs(self):
        return [
            paddle.uniform([171, 84], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_9a52905ea4d1ceafc8927d22e04ee019(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 60, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_38686e4211ddb2fa10fc19d11891df27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a52905ea4d1ceafc8927d22e04ee019
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_3c5047832242eaf2936e08bceb2f12c5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 20, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9ea6e0ba5ddb14b31a2870f81ccbfb0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c5047832242eaf2936e08bceb2f12c5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.823836326599121]], [[4.841972351074219]], [[4.305678844451904]], [[4.914802551269531]], [[5.130475044250488]], [[4.788827896118164]], [[5.480341911315918]], [[5.343693733215332]], [[5.133093357086182]], [[5.910193920135498]], [[5.054744243621826]], [[4.827784538269043]], [[4.356714248657227]], [[4.685656547546387]], [[5.108277320861816]], [[5.094736099243164]], [[5.665359020233154]], [[5.948004245758057]], [[5.116445541381836]], [[4.770357608795166]]]], dtype='float32').reshape([1, 20, 1, 1]),
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
class TestPrimitiveOp_08b5ee3e2b59ca832e99f40da7887bb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf5d9f0bba7a7b73e7e38ae50c877229
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.93044900894165]], [[4.609947204589844]], [[5.418363094329834]], [[5.954451084136963]], [[4.70367431640625]], [[4.194159984588623]], [[5.230362415313721]], [[4.139449596405029]], [[5.085805892944336]], [[4.790792465209961]], [[5.196610450744629]], [[4.634404182434082]], [[4.593898296356201]], [[5.541975498199463]], [[5.049737930297852]], [[4.953497886657715]], [[4.082186698913574]], [[4.560928821563721]]]], dtype='float32').reshape([1, 18, 1, 1]),
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
class TestPrimitiveOp_317ae5ce1a143b32e1ac921d709f50f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_015f60f0f7bb453319399c250296ff51
    def get_inputs(self):
        return [
            paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ac87e20fd8cf8a217bb71b81ad412950(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97a676267ac0217e9a726cdb750b02d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 7, 10], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_11e50181806e25ed31483467fd1c390b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_558de5091dff68c1a5469940856324e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11e50181806e25ed31483467fd1c390b
    def get_inputs(self):
        return [
            paddle.uniform([43, 96, 109, 109], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_c6faba843a864104047205ddfb8d0bbd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_49c8926fde6572ce7769aad0c8025a2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6faba843a864104047205ddfb8d0bbd
    def get_inputs(self):
        return [
            paddle.uniform([43, 16, 54, 54], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_1f475e3f0b24fba0814821005955a9a2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_65dbf3cb6e3fa2ee533ffbcdc17d4ccb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f475e3f0b24fba0814821005955a9a2
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 54, 54], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_0fa327f50b44f76c66cd032d6d6c4f3f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_43ee4ee2844d9b98dabbbe24ba916497(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0fa327f50b44f76c66cd032d6d6c4f3f
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 54, 54], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_f6858bf5978e53aa44f3e9f4dbf32fd0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_96ce5a2d51d5d84431c4ddbaff801067(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6858bf5978e53aa44f3e9f4dbf32fd0
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 54, 54], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_649ba84948d95c0c868e3ac7f0f8af58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0fa327f50b44f76c66cd032d6d6c4f3f
    def get_inputs(self):
        return [
            paddle.uniform([43, 32, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_dad05cb0729707e54c6c4cfdc7d1b345(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6858bf5978e53aa44f3e9f4dbf32fd0
    def get_inputs(self):
        return [
            paddle.uniform([43, 128, 26, 26], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_5886281400e7eb7ed9c1f84512e93d36(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_557e0f5ada6937c12b9ec389828a8c7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5886281400e7eb7ed9c1f84512e93d36
    def get_inputs(self):
        return [
            paddle.uniform([43, 48, 26, 26], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_75c01ed8ab36b8fc71ddd41014521d31(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9913ffd08a0cf249a453be4cf1ce9058(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75c01ed8ab36b8fc71ddd41014521d31
    def get_inputs(self):
        return [
            paddle.uniform([43, 192, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5f551b4276886ee6710beb28c466eb28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f475e3f0b24fba0814821005955a9a2
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1b15e945fae0158d3d8acd7ae79a32d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_bcb14feaaa752cadb88f950caa1edadb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f475e3f0b24fba0814821005955a9a2
    def get_inputs(self):
        return [
            paddle.uniform([43, 64, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8c09e2b0884e79c8cfbf17afb75ab342(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([43, 256, 12, 12], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_2d91316752f88d173c9cea18bb0ff959(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1000, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_506977c45023c397651b40b7751304bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d91316752f88d173c9cea18bb0ff959
    def get_inputs(self):
        return [
            paddle.uniform([43, 1000, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_cca573590ba58b2385a72ea480afdef0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf5d9f0bba7a7b73e7e38ae50c877229
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.953296184539795]], [[4.5094313621521]], [[4.1247944831848145]], [[5.052865982055664]], [[4.30315637588501]], [[4.862283706665039]], [[3.980379104614258]], [[4.318688869476318]], [[4.388061046600342]], [[4.830044269561768]], [[4.465309143066406]], [[4.359241008758545]], [[4.2101731300354]], [[4.413375377655029]], [[4.822516441345215]], [[4.36255407333374]], [[4.419497489929199]], [[4.613299369812012]]]], dtype='float32').reshape([1, 18, 1, 1]),
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
class TestPrimitiveOp_1b8ec91e93b203ab595cc7072283ad6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80ed7330d369c1f34d273814928826ae
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.908949851989746]], [[7.10638952255249]], [[5.833196640014648]], [[7.675716876983643]], [[6.564254283905029]], [[7.208909034729004]], [[7.60159158706665]], [[6.765165328979492]], [[7.721771240234375]], [[6.488300323486328]], [[6.992899417877197]], [[7.120333194732666]], [[5.930432319641113]], [[7.055731773376465]], [[6.70198917388916]], [[6.966367721557617]], [[5.794203758239746]], [[6.519737243652344]], [[6.744708061218262]], [[6.58433723449707]], [[6.722794532775879]], [[6.882261276245117]], [[6.998544216156006]], [[6.971408367156982]]]], dtype='float32').reshape([1, 24, 1, 1]),
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
class TestPrimitiveOp_4f212d07453be99f73d9197a199be61a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 11, 17], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_396486fbc7ccf09d85789aeb8ee6d2d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf5d9f0bba7a7b73e7e38ae50c877229
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.223539352416992]], [[4.981096267700195]], [[4.280937671661377]], [[5.172879695892334]], [[5.1251115798950195]], [[4.71785306930542]], [[5.362261772155762]], [[5.090798377990723]], [[5.013430595397949]], [[4.217739105224609]], [[4.987013339996338]], [[4.386631965637207]], [[5.021262168884277]], [[4.720359802246094]], [[5.020998954772949]], [[5.547306537628174]], [[5.341071128845215]], [[4.673004150390625]]]], dtype='float32').reshape([1, 18, 1, 1]),
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

class PrimitiveOp_d829e76cd8bc50ed20ee7bc38cf80572(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_369b609ebe2badf39f1378ff69a86e44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d829e76cd8bc50ed20ee7bc38cf80572
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f79012152da1f749f1dbf2666c81c671(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97a676267ac0217e9a726cdb750b02d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 10, 16], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ea3625eab27c9a1665a6c812cd3700a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf5d9f0bba7a7b73e7e38ae50c877229
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.273597240447998]], [[3.9308645725250244]], [[4.461357116699219]], [[4.238923072814941]], [[4.377993583679199]], [[4.115373611450195]], [[4.863597393035889]], [[4.693233966827393]], [[4.529014587402344]], [[4.364161968231201]], [[4.21715784072876]], [[4.133312225341797]], [[4.583900451660156]], [[4.583021640777588]], [[4.651145935058594]], [[4.6100592613220215]], [[4.974761009216309]], [[4.1594977378845215]]]], dtype='float32').reshape([1, 18, 1, 1]),
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

class PrimitiveOp_238208547838856b8d51479a53e69c6d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 9], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c9e3f9d6cdbf921b94ecfd7132d886f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_238208547838856b8d51479a53e69c6d
    def get_inputs(self):
        return [
            paddle.uniform([10, 9], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f8ef500aa46c71018286eee3eafe15a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97a676267ac0217e9a726cdb750b02d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 12, 18], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_41ba7185bd9fa5d8fa09975ae7515fe0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11e50181806e25ed31483467fd1c390b
    def get_inputs(self):
        return [
            paddle.uniform([10, 96, 109, 109], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_058646190e08d9119017bb5f46afd5f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6faba843a864104047205ddfb8d0bbd
    def get_inputs(self):
        return [
            paddle.uniform([10, 16, 54, 54], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_93a38d392242259363cc626f53efd5d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f475e3f0b24fba0814821005955a9a2
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 54, 54], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_98a674728c4a63a11269606898602811(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0fa327f50b44f76c66cd032d6d6c4f3f
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 54, 54], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c74843bdeeb299f31de6724b59c10094(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6858bf5978e53aa44f3e9f4dbf32fd0
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 54, 54], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a16f9b13b084ea76b2dd335a5a6b14f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0fa327f50b44f76c66cd032d6d6c4f3f
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_2acbe6d70819f6e710bb62ef1458bfb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6858bf5978e53aa44f3e9f4dbf32fd0
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_37f91790a08b7d4f6407bdce67c66612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5886281400e7eb7ed9c1f84512e93d36
    def get_inputs(self):
        return [
            paddle.uniform([10, 48, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5ca739a7ad1a3a34eb1e356b173e5ac5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75c01ed8ab36b8fc71ddd41014521d31
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_198d4c48309b2e59d3f0e5fa579db7e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f475e3f0b24fba0814821005955a9a2
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_698f5aa4638e1a6d7458ef5ad6677dad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a2ad5a59b52e98d9ec65e3c0d5e8ce29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f475e3f0b24fba0814821005955a9a2
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_db7aa3318d96bfafb12628739f41435b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_eec751309aaaf735c626809184d69df7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d91316752f88d173c9cea18bb0ff959
    def get_inputs(self):
        return [
            paddle.uniform([10, 1000, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9793a83985a5545bcc7f86f6ff54b144(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d787b651e173fdd72b62c54eb2369a9
    def get_inputs(self):
        return [
            paddle.uniform([10, 120], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b75d034f5b719d4e1e0d7f77bdcda92c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262ba2151df345eb3c5ae0bd348a8026
    def get_inputs(self):
        return [
            paddle.uniform([22, 84], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a19371a019153eeebfcc1c2002fdb451(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97a676267ac0217e9a726cdb750b02d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 92, 140], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8da008d522d7895fa66a5261d9ef4e58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_93beaed7fdc06041821f9a131a281c1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_015f60f0f7bb453319399c250296ff51
    def get_inputs(self):
        return [
            paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9bb2764386a75c36ad12f7d8c3290f16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f475e3f0b24fba0814821005955a9a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 300, 300], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f81e8256de2c28efc0e7fcb893a4059d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6858bf5978e53aa44f3e9f4dbf32fd0
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 150, 150], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1e1590cae79824c040f2f01766bb8510(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_b40cf8fc93480e1e685d24ce493ca7b0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9d0583346cab4964d511f7bea2c765c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40cf8fc93480e1e685d24ce493ca7b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_497c5e27c889d11ca71ed736c31a7184(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40cf8fc93480e1e685d24ce493ca7b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_24a5e5ec0efd252971fc110b91e7dd9d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1024, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_32fd52216a73956c5b63e51534975c4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24a5e5ec0efd252971fc110b91e7dd9d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 19, 19], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_487403fd912625ce73453cdfb450204f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 19, 19], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_45425b1b0a86fbca4b3e7ccd6185163d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40cf8fc93480e1e685d24ce493ca7b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 10, 10], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c182c57e5715cc6886c3c6d56c1697ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6858bf5978e53aa44f3e9f4dbf32fd0
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 10, 10], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b92c609e107c5ccffc32e9669f73b874(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 5, 5], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ff20e0fce4938185b1779050269a39de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6858bf5978e53aa44f3e9f4dbf32fd0
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 5, 5], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_0702b4c9774d20736f7f9f999bad9044(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c73a7b1021ac3b285b10600f9e8b0664(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6858bf5978e53aa44f3e9f4dbf32fd0
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 3, 3], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8c4b1be2fd0af04f55cd1b59df80c92d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9e11e636c0b45a46f33f921557cc3481(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97a676267ac0217e9a726cdb750b02d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 13, 19], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_fb2be2969a9b9760ba8e8e05172f0e24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbde9be5291cab7a77e92af8048f17ec
    def get_inputs(self):
        return [
            paddle.uniform([171, 15], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9cc0a6ff1a3694162374c0eff0b20b79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_015f60f0f7bb453319399c250296ff51
    def get_inputs(self):
        return [
            paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_442b684365dc084debd9943b580299ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97a676267ac0217e9a726cdb750b02d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 25, 38], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_2b3d790dcb4a2a3aa28f3591724dc5ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf5d9f0bba7a7b73e7e38ae50c877229
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.113353729248047]], [[5.495994567871094]], [[5.71132755279541]], [[6.018656253814697]], [[4.675065994262695]], [[5.014891147613525]], [[4.965630054473877]], [[4.973498344421387]], [[6.757758140563965]], [[4.501132488250732]], [[5.846304893493652]], [[6.218576431274414]], [[5.281675338745117]], [[5.208498954772949]], [[5.148212432861328]], [[5.077774524688721]], [[5.3781046867370605]], [[4.8508477210998535]]]], dtype='float32').reshape([1, 18, 1, 1]),
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

class PrimitiveOp_e4ade6b2ee6ba32658f59d6f8e8743cc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 13, 19], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a74c6d57d87f24f14bca612b6b95d7c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4ade6b2ee6ba32658f59d6f8e8743cc
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 13, 19], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_632f63acbb546180c3fa6ed431c2d012(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_63d4a3468b5ad3c0789d90cc0d4e4b9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a30ddc94d09c5f2cfeb153b79834946
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.057794570922852]], [[4.007081031799316]], [[4.410943031311035]], [[3.929718255996704]], [[4.576589584350586]], [[4.156850814819336]], [[4.325844764709473]], [[4.249763011932373]], [[4.070028305053711]], [[3.645435094833374]], [[3.646209239959717]], [[4.158534526824951]], [[4.211146354675293]], [[4.155292987823486]], [[4.120391368865967]], [[4.319928169250488]]]], dtype='float32').reshape([1, 16, 1, 1]),
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
class TestPrimitiveOp_9a71067182d1fb86582318724cb2b9c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_238208547838856b8d51479a53e69c6d
    def get_inputs(self):
        return [
            paddle.uniform([22, 9], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_729be529dcbfe8115dac4d48ffc56484(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf5d9f0bba7a7b73e7e38ae50c877229
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.6342549324035645]], [[4.274336338043213]], [[4.769364356994629]], [[5.671443939208984]], [[4.562361240386963]], [[5.569528102874756]], [[5.409242153167725]], [[5.292044639587402]], [[5.178565979003906]], [[5.39305305480957]], [[4.9184956550598145]], [[4.815407752990723]], [[5.106956481933594]], [[5.396690845489502]], [[4.686182975769043]], [[4.850823879241943]], [[5.380630016326904]], [[4.1114678382873535]]]], dtype='float32').reshape([1, 18, 1, 1]),
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
class TestPrimitiveOp_98c24361c545a55564aea50c3d237a8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_474a628119c59d06cc8fb2eb2a66ddb2
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.5894896984100342]], [[1.3004567623138428]], [[1.49582040309906]], [[1.4133695363998413]]]], dtype='float32').reshape([1, 4, 1, 1]),
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
class TestPrimitiveOp_c09885001a3898b40360d1e3d17ca71b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11e50181806e25ed31483467fd1c390b
    def get_inputs(self):
        return [
            paddle.uniform([11, 96, 109, 109], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1629b82f9b9d264df59e57e0310df544(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6faba843a864104047205ddfb8d0bbd
    def get_inputs(self):
        return [
            paddle.uniform([11, 16, 54, 54], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_28845c868806081964769ba364d6a258(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f475e3f0b24fba0814821005955a9a2
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 54, 54], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_51f6d1f6a75dfc26bd6e000f0483347a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0fa327f50b44f76c66cd032d6d6c4f3f
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 54, 54], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_93bab211153fc9a3f5aa72a859b07fca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6858bf5978e53aa44f3e9f4dbf32fd0
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 54, 54], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5aa80fe33952923fed049f8e7b4e0674(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0fa327f50b44f76c66cd032d6d6c4f3f
    def get_inputs(self):
        return [
            paddle.uniform([11, 32, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_493220f4d963512b494ce8e943c2975b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6858bf5978e53aa44f3e9f4dbf32fd0
    def get_inputs(self):
        return [
            paddle.uniform([11, 128, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d183eb4a30dad0de704fac334d8b635e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5886281400e7eb7ed9c1f84512e93d36
    def get_inputs(self):
        return [
            paddle.uniform([11, 48, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_e9c86937873075c216681670b3e723ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75c01ed8ab36b8fc71ddd41014521d31
    def get_inputs(self):
        return [
            paddle.uniform([11, 192, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_e788072bbb64fe4659b4c757729a032d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f475e3f0b24fba0814821005955a9a2
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8944ed069b6fa8021c8b68a6ce5c421d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a35fa6e16806273f49b7b008d6c10065(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f475e3f0b24fba0814821005955a9a2
    def get_inputs(self):
        return [
            paddle.uniform([11, 64, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_3927e24ee32d3a21801ee19b7c9e808f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([11, 256, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_998531eb6224066907af72dcfe410c09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d91316752f88d173c9cea18bb0ff959
    def get_inputs(self):
        return [
            paddle.uniform([11, 1000, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_e3ec8b75aa7b4bbc942d38f923bc742d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbde9be5291cab7a77e92af8048f17ec
    def get_inputs(self):
        return [
            paddle.uniform([145, 15], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_78c97351a28248394bc67eb13bf250f7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 168], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_456e5d883195310ae232b35385bd8fc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_78c97351a28248394bc67eb13bf250f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 168], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_72fe2a48ee5468f7c5c2d180a930b506(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 100, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_93e2d925a5b151abf25605c695763c94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72fe2a48ee5468f7c5c2d180a930b506
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_3b442172412b29e55d4d5f4c0f7625dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c5047832242eaf2936e08bceb2f12c5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.70921516418457]], [[4.715009689331055]], [[5.236291408538818]], [[4.9092183113098145]], [[4.383486747741699]], [[4.950136184692383]], [[5.096542835235596]], [[4.879143714904785]], [[4.668126106262207]], [[5.038344383239746]], [[4.926565170288086]], [[5.196986198425293]], [[4.549531936645508]], [[4.688711643218994]], [[5.383689880371094]], [[5.174729824066162]], [[5.0076518058776855]], [[5.199132442474365]], [[4.6433563232421875]], [[4.807617664337158]]]], dtype='float32').reshape([1, 20, 1, 1]),
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

class PrimitiveOp_f7c9e485d398baed71cdc1ede6777d74(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 84, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f901dd80ed806474c7409d5ce2a5618d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7c9e485d398baed71cdc1ede6777d74
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 1, 1], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_0aaf3f9c9f43c1b89281b8ed63f65d42(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 12, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_917c878390c717093f727e5f6e3f004b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0aaf3f9c9f43c1b89281b8ed63f65d42
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.0434675216674805]], [[3.0076746940612793]], [[4.3080973625183105]], [[3.0539610385894775]], [[4.140078544616699]], [[3.647865056991577]], [[3.7724828720092773]], [[2.946160078048706]], [[2.7036635875701904]], [[2.934163808822632]], [[3.413623094558716]], [[3.533562183380127]]]], dtype='float32').reshape([1, 12, 1, 1]),
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
class TestPrimitiveOp_431dc58882ff87294a9d411db2587f5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c5047832242eaf2936e08bceb2f12c5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.164685249328613]], [[5.6693549156188965]], [[4.820473670959473]], [[5.086769104003906]], [[5.46857213973999]], [[4.941510200500488]], [[5.570308685302734]], [[5.455769062042236]], [[4.548212051391602]], [[5.573060035705566]], [[4.531099319458008]], [[5.059079647064209]], [[5.089148044586182]], [[5.488116264343262]], [[5.749452114105225]], [[4.732754707336426]], [[5.111810684204102]], [[5.471380233764648]], [[4.567051410675049]], [[5.362548828125]]]], dtype='float32').reshape([1, 20, 1, 1]),
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
class TestPrimitiveOp_fcce79bb801a90a114c78d311c0058a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d0a332f3f65a0baa0b24c0fb0a72a04c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.554349660873413]], [[2.984191417694092]], [[3.0773959159851074]], [[2.9575600624084473]], [[2.2049384117126465]], [[3.0560779571533203]], [[2.0327553749084473]], [[3.163947343826294]], [[2.833745241165161]], [[2.7105236053466797]], [[1.7433278560638428]]]], dtype='float32').reshape([1, 11, 1, 1]),
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
class TestPrimitiveOp_efb35a7c332a288b46efb0bbf27af76b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97a676267ac0217e9a726cdb750b02d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 56, 80], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_20328261b99abcac38b3c71db66df47f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 14, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9fe137d1e6d54cf5b08ab46f1d3ca490(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20328261b99abcac38b3c71db66df47f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.414215087890625]], [[3.9895942211151123]], [[3.3893890380859375]], [[3.991058349609375]], [[3.3325841426849365]], [[2.972386360168457]], [[4.130126953125]], [[3.263284921646118]], [[3.8213534355163574]], [[3.9842443466186523]], [[3.1546404361724854]], [[3.424934148788452]], [[3.3857316970825195]], [[3.1984293460845947]]]], dtype='float32').reshape([1, 14, 1, 1]),
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

class PrimitiveOp_85ed9acc28e00af511e7803f56e3294e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d4bbff5adc6304be723ac83be663b66d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85ed9acc28e00af511e7803f56e3294e
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ac08a66d640f8c9c1a9d2f348d5b329f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c5047832242eaf2936e08bceb2f12c5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.825429439544678]], [[4.922501087188721]], [[4.990355968475342]], [[5.489058494567871]], [[4.323785305023193]], [[4.699490547180176]], [[5.001468658447266]], [[4.936253070831299]], [[4.6549482345581055]], [[4.829311370849609]], [[5.099071025848389]], [[4.964115142822266]], [[4.749936103820801]], [[4.554876804351807]], [[4.3140411376953125]], [[4.654892444610596]], [[5.267151355743408]], [[4.372352123260498]], [[5.095065593719482]], [[4.78563928604126]]]], dtype='float32').reshape([1, 20, 1, 1]),
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
class TestPrimitiveOp_04acc47440cb564ebe3112e01d7cb447(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80ed7330d369c1f34d273814928826ae
    def get_inputs(self):
        return [
            paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_8276864881ce911fc9b794b28454fd15(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 6, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_447e5ab440d6fdad96a5cb4a9f4f3ef8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8276864881ce911fc9b794b28454fd15
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[35703.07421875]], [[29847.548828125]], [[35764.2890625]], [[41210.7734375]], [[28388.34765625]], [[38335.7578125]]], [[[35184.9375]], [[29422.140625]], [[35248.55859375]], [[40622.1796875]], [[27982.34375]], [[37782.7890625]]]], dtype='float32').reshape([2, 6, 1, 1]),
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
class TestPrimitiveOp_f9f98c242137b238dc5825ce2e929958(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8276864881ce911fc9b794b28454fd15
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[36232.578125]], [[32477.4140625]], [[38839.375]], [[36755.08984375]], [[48853.46875]], [[38602.6171875]]], [[[35125.6171875]], [[31485.537109375]], [[37655.41796875]], [[35633.1875]], [[47361.4296875]], [[37426.2265625]]]], dtype='float32').reshape([2, 6, 1, 1]),
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
class TestPrimitiveOp_baaa36c821e5a5344f3ef5c87e4ff977(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8276864881ce911fc9b794b28454fd15
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[40847.02734375]], [[40062.1953125]], [[45187.515625]], [[47650.5703125]], [[41011.328125]], [[38968.39453125]]], [[[39265.29296875]], [[38511.7109375]], [[43442.58984375]], [[45809.91015625]], [[39421.35546875]], [[37462.34375]]]], dtype='float32').reshape([2, 6, 1, 1]),
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
class TestPrimitiveOp_33a718db8621935c2dc38450053c0aad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8276864881ce911fc9b794b28454fd15
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[37322.234375]], [[45104.25390625]], [[46323.05859375]], [[44561.17578125]], [[40449.51953125]], [[36808.828125]]], [[[36430.921875]], [[44031.66796875]], [[45224.703125]], [[43501.35546875]], [[39489.7109375]], [[35936.8515625]]]], dtype='float32').reshape([2, 6, 1, 1]),
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
class TestPrimitiveOp_625e39569a696b73a64825395d4207a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_744a4862f6ed7685aa950d1b2266fc39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4586088d9a8789706d22c80ac8e24e05(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8c64e8dcf88a79190647d5a5345c22da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4a7adec38b7b1fe438cc3d156cc6aa70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54278e400b2be87ca05c602cb1ca8126
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.6658353805542]], [[8.1070556640625]], [[8.222827911376953]], [[7.877943992614746]], [[8.066951751708984]], [[7.292673110961914]], [[8.084744453430176]], [[7.5229811668396]], [[8.21318244934082]], [[7.604622840881348]], [[8.83447265625]], [[8.344555854797363]], [[9.45718002319336]], [[7.1796956062316895]], [[8.142525672912598]], [[7.744045734405518]], [[7.552488803863525]], [[7.6111650466918945]], [[7.037825107574463]], [[9.103200912475586]], [[8.932579040527344]], [[8.454848289489746]], [[7.616219997406006]], [[8.000482559204102]], [[7.608373165130615]], [[7.816816806793213]], [[8.14023494720459]], [[8.14525318145752]], [[7.936910629272461]], [[8.751760482788086]]]], dtype='float32').reshape([1, 30, 1, 1]),
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
class TestPrimitiveOp_9263c52227faf524f2a082a614a93bbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54278e400b2be87ca05c602cb1ca8126
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.416786193847656]], [[7.794103622436523]], [[8.924464225769043]], [[7.228585243225098]], [[9.031545639038086]], [[8.148994445800781]], [[7.675243854522705]], [[7.80863618850708]], [[7.688236236572266]], [[8.47314453125]], [[7.7580342292785645]], [[7.796664714813232]], [[7.8048248291015625]], [[8.402878761291504]], [[7.834222316741943]], [[7.293983459472656]], [[7.622461795806885]], [[8.414867401123047]], [[8.357909202575684]], [[8.451399803161621]], [[6.94551944732666]], [[8.028322219848633]], [[7.767489910125732]], [[8.670419692993164]], [[7.389835357666016]], [[7.984164237976074]], [[8.085855484008789]], [[7.575388431549072]], [[7.201699733734131]], [[8.10418701171875]]]], dtype='float32').reshape([1, 30, 1, 1]),
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
class TestPrimitiveOp_2fc0dd9dbac1e8cc69b8501f2201fa4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97a676267ac0217e9a726cdb750b02d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 44, 66], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c55ecd661560b528d5c8131a01cfb73b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54278e400b2be87ca05c602cb1ca8126
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.368825435638428]], [[7.740781307220459]], [[8.700937271118164]], [[8.02814769744873]], [[7.299408435821533]], [[7.574045181274414]], [[7.590500831604004]], [[7.792178630828857]], [[7.857649803161621]], [[7.405430793762207]], [[7.992948055267334]], [[7.77955436706543]], [[7.631563186645508]], [[7.997115135192871]], [[7.894710540771484]], [[7.477319240570068]], [[7.802051544189453]], [[7.672013282775879]], [[7.112525463104248]], [[7.324472904205322]], [[7.04571008682251]], [[8.032870292663574]], [[7.49959135055542]], [[8.095975875854492]], [[7.526414394378662]], [[8.05590534210205]], [[8.160518646240234]], [[8.33193302154541]], [[8.011821746826172]], [[7.693360328674316]]]], dtype='float32').reshape([1, 30, 1, 1]),
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

class PrimitiveOp_df290d4dd1ff071aa1f7aea637728d4c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 50, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c1a7716f96566d28da910415e365a82d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df290d4dd1ff071aa1f7aea637728d4c
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_68da49cbffa22c052f7b0ee8f406efef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54278e400b2be87ca05c602cb1ca8126
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.7651166915893555]], [[7.80988883972168]], [[7.753073215484619]], [[7.550171852111816]], [[7.778337001800537]], [[7.513052463531494]], [[6.393861770629883]], [[8.28150749206543]], [[7.846116542816162]], [[7.819559097290039]], [[8.35317611694336]], [[8.397684097290039]], [[7.938183307647705]], [[7.887857437133789]], [[7.237251281738281]], [[7.872225284576416]], [[8.066829681396484]], [[8.189265251159668]], [[8.395303726196289]], [[8.470483779907227]], [[8.274739265441895]], [[8.872918128967285]], [[6.792125701904297]], [[7.824231147766113]], [[8.335457801818848]], [[7.598811626434326]], [[7.617952346801758]], [[8.056575775146484]], [[7.427498817443848]], [[8.475411415100098]]]], dtype='float32').reshape([1, 30, 1, 1]),
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
class TestPrimitiveOp_d8766a7c02dff6d86b1210c5162a7311(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0aaf3f9c9f43c1b89281b8ed63f65d42
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.968533754348755]], [[3.2744359970092773]], [[4.22754430770874]], [[3.6164140701293945]], [[3.2631449699401855]], [[3.6471145153045654]], [[3.716449737548828]], [[3.3851146697998047]], [[3.310685634613037]], [[3.4721317291259766]], [[2.960721254348755]], [[3.1858975887298584]]]], dtype='float32').reshape([1, 12, 1, 1]),
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
class TestPrimitiveOp_e831a45a19fc055bafb775a070febec5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0aaf3f9c9f43c1b89281b8ed63f65d42
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.6957502365112305]], [[2.9693329334259033]], [[2.8324947357177734]], [[3.0528671741485596]], [[2.9085195064544678]], [[3.041714668273926]], [[2.9788265228271484]], [[3.414797306060791]], [[3.3351552486419678]], [[3.358827590942383]], [[3.3882534503936768]], [[3.147059440612793]]]], dtype='float32').reshape([1, 12, 1, 1]),
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
class TestPrimitiveOp_0295c1684bc84370ebc26e705c5769a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ad8d2f145fdbb6389a6e940b3876077
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.293083190917969]], [[5.870551109313965]], [[6.320681571960449]], [[6.907689571380615]], [[6.797855377197266]], [[6.495174407958984]], [[5.911848068237305]], [[6.560781955718994]], [[6.150582313537598]], [[6.502373218536377]], [[6.399684429168701]], [[5.962441921234131]], [[6.735020637512207]], [[7.271669387817383]], [[6.681186199188232]], [[6.115781307220459]], [[6.6929850578308105]], [[7.5167646408081055]], [[6.415417671203613]], [[6.338051795959473]], [[6.1207966804504395]], [[6.665828704833984]], [[6.196653366088867]], [[6.108193397521973]], [[7.369046211242676]]]], dtype='float32').reshape([1, 25, 1, 1]),
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

class PrimitiveOp_9eb87c108dfd0ea1413bcaee6ed90ca5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 72, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0d44a17065b8be59d3d34df1bf757fcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9eb87c108dfd0ea1413bcaee6ed90ca5
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_ce50b3fa1726b6e72affe2211901b20f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 312], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ccd7454861fdc5e926b48d6a977f8769(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce50b3fa1726b6e72affe2211901b20f
    def get_inputs(self):
        return [
            paddle.uniform([1, 312], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c012659453ddd863292dae97cf000833(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d787b651e173fdd72b62c54eb2369a9
    def get_inputs(self):
        return [
            paddle.uniform([171, 120], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_fc22259e7ac5ff82d98f75a31435dcc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_238208547838856b8d51479a53e69c6d
    def get_inputs(self):
        return [
            paddle.uniform([145, 9], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_e0f90de90bea669139bf22e944e6415f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97a676267ac0217e9a726cdb750b02d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 5, 8], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9e6b576120588f7affc5f2700f7107bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf5d9f0bba7a7b73e7e38ae50c877229
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.721285343170166]], [[4.808663368225098]], [[3.7746777534484863]], [[4.069631576538086]], [[3.8913345336914062]], [[4.651084899902344]], [[4.408143520355225]], [[3.8639371395111084]], [[4.055018424987793]], [[3.73452091217041]], [[4.03886604309082]], [[3.93035626411438]], [[4.237965106964111]], [[4.525489330291748]], [[4.1659650802612305]], [[3.751066207885742]], [[3.5675864219665527]], [[3.557084083557129]]]], dtype='float32').reshape([1, 18, 1, 1]),
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

class PrimitiveOp_b6e368c9ff81158ef8cbd6e1e50e8a3e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 39], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1536a1c357d5dbc985b13604e631985b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6e368c9ff81158ef8cbd6e1e50e8a3e
    def get_inputs(self):
        return [
            paddle.uniform([1, 39], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_f18a90c697362949c353e010cdf07ae5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 5, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_63f0f59f3e21866f0c8527720599aa13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f18a90c697362949c353e010cdf07ae5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.8998667001724243]], [[1.3319482803344727]], [[1.4812557697296143]], [[1.1986289024353027]], [[1.647526502609253]]]], dtype='float32').reshape([1, 5, 1, 1]),
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

class PrimitiveOp_7b04b99ae0d930a294a8dcfea28d0756(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 10, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d70bc46ebda76684c438e7622abbeb27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b04b99ae0d930a294a8dcfea28d0756
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.685392141342163]], [[3.432683229446411]], [[2.8449554443359375]], [[2.8167319297790527]], [[2.9428277015686035]], [[2.992644786834717]], [[2.637897491455078]], [[2.3180813789367676]], [[2.680163860321045]], [[3.168722629547119]]]], dtype='float32').reshape([1, 10, 1, 1]),
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

class PrimitiveOp_074b2732436b8fcd52cbf71478d86508(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 20, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3c99c44f2b50d83222516c8b19ab42b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_074b2732436b8fcd52cbf71478d86508
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.834033489227295]], [[6.028222560882568]], [[6.2730607986450195]], [[5.815683364868164]], [[5.537166595458984]], [[5.755615234375]], [[5.125760078430176]], [[4.9682207107543945]], [[5.588512897491455]], [[6.035160064697266]], [[4.70105504989624]], [[5.620141506195068]], [[5.955061912536621]], [[5.09976863861084]], [[5.872868537902832]], [[6.469480991363525]], [[5.581788539886475]], [[5.994884967803955]], [[5.6630120277404785]], [[5.77250337600708]]]], dtype='float32').reshape([1, 20, 1, 1]),
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

class PrimitiveOp_0ac3017314973ad892428c9f1d8ed87c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 40, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ba0a64ac96dc54cc894e82ad77332fe5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ac3017314973ad892428c9f1d8ed87c
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_990294bc1594bed08d2f67d1d8222cb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80ed7330d369c1f34d273814928826ae
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.5755462646484375]], [[7.724958419799805]], [[6.630558490753174]], [[7.034902572631836]], [[6.702817916870117]], [[6.833985328674316]], [[6.586999416351318]], [[6.8699631690979]], [[6.244996547698975]], [[6.929019451141357]], [[7.096578598022461]], [[6.641002178192139]], [[6.750032901763916]], [[6.835293769836426]], [[6.002835273742676]], [[7.5838422775268555]], [[7.019974708557129]], [[6.836799621582031]], [[6.463237285614014]], [[7.26837682723999]], [[6.874286651611328]], [[7.179231643676758]], [[6.164690017700195]], [[7.2050933837890625]]]], dtype='float32').reshape([1, 24, 1, 1]),
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
class TestPrimitiveOp_0d7471684e65a142b61db767e0eb2fca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d787b651e173fdd72b62c54eb2369a9
    def get_inputs(self):
        return [
            paddle.uniform([22, 120], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1f5bd5f0345553172c80e04b6d198ede(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db3a420cbd0dbb845b550b1a09e56109
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.720344066619873]], [[2.8081295490264893]], [[2.952676296234131]], [[2.218912124633789]], [[2.8770737648010254]], [[2.590301990509033]], [[2.5677194595336914]], [[2.233036518096924]], [[2.099064350128174]], [[2.477353096008301]]]], dtype='float32').reshape([1, 10, 1, 1]),
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
class TestPrimitiveOp_a53ea823f9279fe86c8a37efcbdafb08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d787b651e173fdd72b62c54eb2369a9
    def get_inputs(self):
        return [
            paddle.uniform([145, 120], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_aeeb0e7c45c34800b0132ebe1f2004b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97a676267ac0217e9a726cdb750b02d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 40, 64], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_0a995de933c01e5e6a5ec0f2ce5a554b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 50, 76], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_191bb5793d3356ed24222f2758fb36f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a995de933c01e5e6a5ec0f2ce5a554b
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 50, 76], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_972b96c27bb2638b4b8a232ff0b89632(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_238208547838856b8d51479a53e69c6d
    def get_inputs(self):
        return [
            paddle.uniform([171, 9], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_666f9b2f399669ca3d321f0cc129303e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf5d9f0bba7a7b73e7e38ae50c877229
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.405776023864746]], [[4.127753257751465]], [[4.362503528594971]], [[4.529914855957031]], [[4.277790069580078]], [[4.447083473205566]], [[4.63477897644043]], [[4.138066291809082]], [[3.7077553272247314]], [[5.377406120300293]], [[3.928138494491577]], [[4.060510635375977]], [[4.17971134185791]], [[4.1579790115356445]], [[3.9366140365600586]], [[4.897757053375244]], [[4.514909267425537]], [[4.128507137298584]]]], dtype='float32').reshape([1, 18, 1, 1]),
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

class PrimitiveOp_ee3bc7956b0e79210390fb0751d164e9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 30], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d82ea996943a987318eb04d7b87b5f33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee3bc7956b0e79210390fb0751d164e9
    def get_inputs(self):
        return [
            paddle.to_tensor([[8.378665924072266, 8.622936248779297, 7.992419242858887, 7.285286903381348, 7.504433631896973, 7.695581912994385, 7.320318698883057, 7.855299949645996, 7.528284072875977, 7.803767681121826, 8.484917640686035, 8.615951538085938, 7.278975009918213, 7.6329498291015625, 7.102418422698975, 7.5837626457214355, 7.093390941619873, 8.706185340881348, 8.39137077331543, 7.464969158172607, 8.560622215270996, 8.868217468261719, 7.718655586242676, 8.08227825164795, 7.667810440063477, 7.491669178009033, 7.475050449371338, 7.9493088722229, 8.244874000549316, 7.5856757164001465]], dtype='float32').reshape([1, 30]),
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
class TestPrimitiveOp_37699b4f716daba4062beba05956ac0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54278e400b2be87ca05c602cb1ca8126
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.708158016204834]], [[7.252604961395264]], [[7.377367973327637]], [[6.789035320281982]], [[7.606326580047607]], [[7.113602638244629]], [[6.4286789894104]], [[6.917911529541016]], [[7.2128777503967285]], [[7.766190528869629]], [[6.791611194610596]], [[6.892510890960693]], [[6.806152820587158]], [[7.006817817687988]], [[7.4687395095825195]], [[5.781470775604248]], [[6.6284332275390625]], [[7.578555107116699]], [[6.092510223388672]], [[7.131056308746338]], [[5.506556034088135]], [[6.615859031677246]], [[6.5423197746276855]], [[6.561517238616943]], [[6.8081374168396]], [[6.281248092651367]], [[7.557594299316406]], [[6.767190456390381]], [[7.261711120605469]], [[6.651391983032227]]]], dtype='float32').reshape([1, 30, 1, 1]),
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
class TestPrimitiveOp_724b96fab04311f6c2a4ec5ccc844ccb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ccabd4bdc8c15e1883d6b58f2ffcf4ee
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.7736645936965942]], [[1.3818273544311523]], [[1.6397769451141357]], [[1.2828425168991089]], [[1.889788031578064]]]], dtype='float32').reshape([1, 5, 1, 1]),
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
class TestPrimitiveOp_6e9e874afa53aee70de326a6e351550d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db3a420cbd0dbb845b550b1a09e56109
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.642515182495117]], [[2.7344796657562256]], [[2.4162209033966064]], [[3.6685612201690674]], [[2.6550488471984863]], [[2.7155582904815674]], [[2.9602038860321045]], [[3.0387861728668213]], [[3.186140775680542]], [[3.3328189849853516]]]], dtype='float32').reshape([1, 10, 1, 1]),
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
class TestPrimitiveOp_c6ee8f6e7ac5b3646da58978a7500c47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c5047832242eaf2936e08bceb2f12c5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.030589580535889]], [[5.035013675689697]], [[4.420341491699219]], [[5.301993370056152]], [[5.071216106414795]], [[5.969995021820068]], [[5.171087265014648]], [[6.536515235900879]], [[5.798988342285156]], [[5.952205181121826]], [[6.107416152954102]], [[6.40071964263916]], [[5.373867034912109]], [[5.728961944580078]], [[5.953717231750488]], [[5.541092395782471]], [[5.195215225219727]], [[6.765068054199219]], [[5.535505771636963]], [[4.950847148895264]]]], dtype='float32').reshape([1, 20, 1, 1]),
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
class TestPrimitiveOp_c38f6af883da95d9f4d343722fc36e13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a30ddc94d09c5f2cfeb153b79834946
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.375254154205322]], [[3.876783609390259]], [[4.515564441680908]], [[4.219099998474121]], [[4.444534778594971]], [[3.815962553024292]], [[4.503503322601318]], [[4.283707618713379]], [[4.291059494018555]], [[4.729404926300049]], [[4.597497940063477]], [[4.576755046844482]], [[4.495851039886475]], [[4.035416603088379]], [[3.8874900341033936]], [[4.451059341430664]]]], dtype='float32').reshape([1, 16, 1, 1]),
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

class PrimitiveOp_c4935bab54bb92d8bc88e7b43ce7babb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 36, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ccba68fb4ce172eb5d8727319e0c701d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4935bab54bb92d8bc88e7b43ce7babb
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ceb207f5b609da299fe8b9d441516f30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5e688cce609933924044c56c35e8f8a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d51bcd2c6815f883990c100b05818fa4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4fdff2a1132e2ddb6191cc3c9b355063(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20328261b99abcac38b3c71db66df47f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.194644927978516]], [[3.3138089179992676]], [[4.333257675170898]], [[3.9553563594818115]], [[4.612022399902344]], [[4.258015155792236]], [[4.4418416023254395]], [[4.1807708740234375]], [[4.294811248779297]], [[4.170912742614746]], [[4.017587184906006]], [[4.550390243530273]], [[3.9489927291870117]], [[3.9343838691711426]]]], dtype='float32').reshape([1, 14, 1, 1]),
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
class TestPrimitiveOp_583aae4096c0f9b63c1d11eec69af368(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c5047832242eaf2936e08bceb2f12c5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.218982219696045]], [[5.918820858001709]], [[5.5988874435424805]], [[4.726998329162598]], [[5.55240535736084]], [[6.374054431915283]], [[5.321900844573975]], [[5.318398475646973]], [[5.078855991363525]], [[5.079309463500977]], [[5.932351112365723]], [[6.2274298667907715]], [[5.890310764312744]], [[5.522193431854248]], [[5.292682647705078]], [[5.201898574829102]], [[5.938455581665039]], [[5.890253067016602]], [[4.985781192779541]], [[5.914712429046631]]]], dtype='float32').reshape([1, 20, 1, 1]),
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
class TestPrimitiveOp_72b643e259a6c1f226af67d7f6b4e2c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 14, 20], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d6f68adcfe7868449e70305d706b8217(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_54278e400b2be87ca05c602cb1ca8126
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.02066421508789]], [[7.951235294342041]], [[7.977724075317383]], [[8.337994575500488]], [[8.380784034729004]], [[9.448760032653809]], [[8.62950325012207]], [[8.02839183807373]], [[8.133667945861816]], [[7.411727428436279]], [[7.980532646179199]], [[8.305061340332031]], [[8.481549263000488]], [[7.9515485763549805]], [[7.594925403594971]], [[7.8966498374938965]], [[8.162572860717773]], [[7.834351062774658]], [[7.796360969543457]], [[7.60296106338501]], [[8.017879486083984]], [[7.912840366363525]], [[7.13515567779541]], [[7.504832744598389]], [[8.09492015838623]], [[7.242308616638184]], [[8.097259521484375]], [[7.662278652191162]], [[8.42585563659668]], [[7.133913993835449]]]], dtype='float32').reshape([1, 30, 1, 1]),
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
class TestPrimitiveOp_28678929183a0bfdd2867fa0faebfd0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11e50181806e25ed31483467fd1c390b
    def get_inputs(self):
        return [
            paddle.uniform([22, 96, 109, 109], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b26501b0fea91a38179e96b2b80df9bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6faba843a864104047205ddfb8d0bbd
    def get_inputs(self):
        return [
            paddle.uniform([22, 16, 54, 54], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_e47353c0d2496753accff26bd12d3045(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f475e3f0b24fba0814821005955a9a2
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 54, 54], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ed9f7824365c6f3982180c82faacc4e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0fa327f50b44f76c66cd032d6d6c4f3f
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 54, 54], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ffaeb78900b871c336f9c645fe5ad069(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6858bf5978e53aa44f3e9f4dbf32fd0
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 54, 54], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_688fb3cd4c1c63ee9ea76746fa942c41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0fa327f50b44f76c66cd032d6d6c4f3f
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_3df121c9e9f0694de0b8b190f1f72db2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f6858bf5978e53aa44f3e9f4dbf32fd0
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_94a16bc93adf8f33118a47722ad15150(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5886281400e7eb7ed9c1f84512e93d36
    def get_inputs(self):
        return [
            paddle.uniform([22, 48, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_06b4cff6458152569a185fa357200ccd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75c01ed8ab36b8fc71ddd41014521d31
    def get_inputs(self):
        return [
            paddle.uniform([22, 192, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8689792eb669c33ff65d82acf42902b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f475e3f0b24fba0814821005955a9a2
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6ec24b63eeb454ad39e888443eede6ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 26, 26], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_01e8b2a2835a04e5ec6aa688dce55286(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f475e3f0b24fba0814821005955a9a2
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_149466f6e1dabef6a9c6a61a8ee0a992(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_51ae22b0a10bd39597d6ee50bb70fad7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d91316752f88d173c9cea18bb0ff959
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000, 12, 12], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c0514de580e73ecfcf3b5d7ddaad2014(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a324f85e300e87f041a35ba8fa34127e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 10, 16], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1b2a660ce0dea62feca6cc058e59defa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80ed7330d369c1f34d273814928826ae
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.767958641052246]], [[6.259011268615723]], [[6.5256195068359375]], [[5.72200345993042]], [[6.045093536376953]], [[6.097092628479004]], [[6.170426368713379]], [[5.24308967590332]], [[5.914750099182129]], [[6.250386714935303]], [[6.673716068267822]], [[6.721867084503174]], [[5.911425590515137]], [[6.5793328285217285]], [[6.046773433685303]], [[6.206075191497803]], [[6.1781110763549805]], [[7.2541093826293945]], [[5.930622577667236]], [[5.391793727874756]], [[5.651362895965576]], [[6.339712619781494]], [[5.226492404937744]], [[5.5235443115234375]]]], dtype='float32').reshape([1, 24, 1, 1]),
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
class TestPrimitiveOp_3a4b642d35b8a60fb818108b3856f3cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ad8d2f145fdbb6389a6e940b3876077
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.254484176635742]], [[6.779487609863281]], [[5.887111186981201]], [[6.078207492828369]], [[7.187652111053467]], [[6.1240434646606445]], [[6.604331016540527]], [[7.094372749328613]], [[6.038271903991699]], [[6.957431793212891]], [[6.0269670486450195]], [[7.332108497619629]], [[6.249767780303955]], [[6.39636754989624]], [[7.155977249145508]], [[6.703110218048096]], [[5.996480941772461]], [[7.105651378631592]], [[5.985050201416016]], [[5.435423851013184]], [[6.244361400604248]], [[5.577609539031982]], [[5.8099589347839355]], [[5.968587875366211]], [[6.123788833618164]]]], dtype='float32').reshape([1, 25, 1, 1]),
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
class TestPrimitiveOp_f32153c823e1ade6448441eab157db37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0aaf3f9c9f43c1b89281b8ed63f65d42
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.888890504837036]], [[3.3929007053375244]], [[2.6112351417541504]], [[3.3341012001037598]], [[2.9060111045837402]], [[3.714590072631836]], [[3.5299689769744873]], [[2.686753511428833]], [[2.849156379699707]], [[3.0124778747558594]], [[2.9774067401885986]], [[3.0496160984039307]]]], dtype='float32').reshape([1, 12, 1, 1]),
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

class PrimitiveOp_db50e9c28683fa08f4de46e72eed11b3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 25, 38], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_16abc52ccd26a46dce50f589e2428f67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_db50e9c28683fa08f4de46e72eed11b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 25, 38], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5a7cffdbffd6a433b04316df2ef247e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97a676267ac0217e9a726cdb750b02d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 112, 160], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_2c6d16199fb6e87c3bef398cba326ac6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 7, 10], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9b3a70689ea256f17c75416145d4a7da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c6d16199fb6e87c3bef398cba326ac6
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 7, 10], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_2e84215d7b1f2f2173abc12a898530ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80ed7330d369c1f34d273814928826ae
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[655.7572021484375]], [[705.666015625]], [[773.6893920898438]], [[709.565673828125]], [[721.9839477539062]], [[814.34814453125]], [[689.0104370117188]], [[783.5064697265625]], [[732.5365600585938]], [[706.2600708007812]], [[773.250244140625]], [[794.0547485351562]], [[746.3873901367188]], [[696.74853515625]], [[749.4446411132812]], [[793.5221557617188]], [[676.5300903320312]], [[676.3561401367188]], [[726.6395263671875]], [[715.7932739257812]], [[653.9281005859375]], [[629.6777954101562]], [[765.410400390625]], [[767.4696655273438]]]], dtype='float32').reshape([1, 24, 1, 1]),
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
class TestPrimitiveOp_e7806465aee510c05e6ba4c285db4e85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80ed7330d369c1f34d273814928826ae
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[74.81359100341797]], [[78.00292205810547]], [[86.67084503173828]], [[90.57637786865234]], [[75.38327026367188]], [[83.65338897705078]], [[88.43961334228516]], [[84.3248062133789]], [[84.56562042236328]], [[86.31048583984375]], [[82.31513977050781]], [[77.43717193603516]], [[79.58155059814453]], [[86.9212417602539]], [[87.23821258544922]], [[82.0116195678711]], [[83.5269775390625]], [[92.50822448730469]], [[90.42761993408203]], [[90.77925872802734]], [[82.18746185302734]], [[81.39210510253906]], [[83.22457885742188]], [[90.01710510253906]]]], dtype='float32').reshape([1, 24, 1, 1]),
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
class TestPrimitiveOp_327cf27fdc32bf8dcdbacacb8801aa47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80ed7330d369c1f34d273814928826ae
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[33.23279571533203]], [[30.137006759643555]], [[29.386131286621094]], [[25.653465270996094]], [[32.61014938354492]], [[26.98526954650879]], [[31.62348175048828]], [[29.61621856689453]], [[31.396163940429688]], [[32.07232666015625]], [[29.105955123901367]], [[30.439414978027344]], [[30.328889846801758]], [[28.05185317993164]], [[31.804729461669922]], [[28.109773635864258]], [[31.42383575439453]], [[29.110925674438477]], [[28.396617889404297]], [[27.790151596069336]], [[26.741432189941406]], [[30.17379379272461]], [[29.696020126342773]], [[33.10212326049805]]]], dtype='float32').reshape([1, 24, 1, 1]),
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
class TestPrimitiveOp_8ee8c6c11378b543f70fa57094801a98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80ed7330d369c1f34d273814928826ae
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[31.59615707397461]], [[32.04358673095703]], [[29.37789535522461]], [[31.304113388061523]], [[31.302127838134766]], [[29.74681282043457]], [[32.08837890625]], [[31.049131393432617]], [[31.736263275146484]], [[29.832345962524414]], [[32.678524017333984]], [[28.981279373168945]], [[30.287275314331055]], [[29.703840255737305]], [[29.753652572631836]], [[28.28385353088379]], [[30.456066131591797]], [[33.913978576660156]], [[31.88187599182129]], [[30.4820556640625]], [[28.127578735351562]], [[30.154258728027344]], [[31.139921188354492]], [[29.98908233642578]]]], dtype='float32').reshape([1, 24, 1, 1]),
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
class TestPrimitiveOp_7ae73ed4929628b1661d3d4483eb9688(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8276864881ce911fc9b794b28454fd15
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[32348.275390625]], [[35888.19921875]], [[37447.94921875]], [[28942.8515625]], [[34149.21484375]], [[28461.111328125]]]], dtype='float32').reshape([1, 6, 1, 1]),
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
class TestPrimitiveOp_b94f1d01b4a3320caa1808b6e1d0abd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8276864881ce911fc9b794b28454fd15
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[39232.3828125]], [[35451.703125]], [[41437.140625]], [[39485.75]], [[39789.12109375]], [[44328.6796875]]]], dtype='float32').reshape([1, 6, 1, 1]),
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
class TestPrimitiveOp_d2fe9118a8d716dac58e305ca288ef6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8276864881ce911fc9b794b28454fd15
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[48054.35546875]], [[46217.55859375]], [[37226.0390625]], [[40559.53125]], [[43859.953125]], [[41513.4921875]]]], dtype='float32').reshape([1, 6, 1, 1]),
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
class TestPrimitiveOp_68ce1a365376db87143f7cae25d09d0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8276864881ce911fc9b794b28454fd15
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[48646.22265625]], [[41030.73046875]], [[43503.50390625]], [[41555.27734375]], [[38020.99609375]], [[36831.12109375]]]], dtype='float32').reshape([1, 6, 1, 1]),
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
class TestPrimitiveOp_d4966d39265d512c979a31facf9a60b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97a676267ac0217e9a726cdb750b02d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 11, 17], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a3e3b46987adf2a39b2ac9900782aacf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97a676267ac0217e9a726cdb750b02d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 88, 132], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f7916f67d50a5e8a29ee9d13782c740c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80ed7330d369c1f34d273814928826ae
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7.077600955963135]], [[6.519475936889648]], [[6.887722015380859]], [[6.376555442810059]], [[6.385010242462158]], [[7.068292140960693]], [[7.098506927490234]], [[5.753015518188477]], [[6.571295738220215]], [[7.231013298034668]], [[7.227197647094727]], [[6.2340474128723145]], [[6.135470867156982]], [[7.156917095184326]], [[6.59360933303833]], [[6.4501142501831055]], [[6.219383716583252]], [[6.524694442749023]], [[7.338897705078125]], [[6.9839887619018555]], [[6.738530158996582]], [[6.914845943450928]], [[7.0328049659729]], [[6.626660346984863]]]], dtype='float32').reshape([1, 24, 1, 1]),
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

class PrimitiveOp_95e8d6e61f483b9084506df27e3f81b2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 100, 152], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5cdfa78bf832b2add94f79c634a9ff77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95e8d6e61f483b9084506df27e3f81b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 100, 152], dtype='float32', min=0, max=0.5),
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

class PrimitiveOp_a4f4df29f08a776807aaca3fcf70d6a0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.relu(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 156], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bb030f8827c8fd4081e5e5416a29f2b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4f4df29f08a776807aaca3fcf70d6a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
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