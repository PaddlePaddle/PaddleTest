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
class PrimitiveOp_9b6593f195462e9032f7f662c64034a0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.unfold(input_0, [3, 3], [1, 1], [1, 1, 1, 1], [1, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b1acdd3be8fdcc6b0610643ac6ab7ae2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b6593f195462e9032f7f662c64034a0
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 112, 112], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_ca6c98e77ee8085ea985376866565a60(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.unfold(input_0, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_11a0c78a823a4e095f3c23474d9773fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca6c98e77ee8085ea985376866565a60
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dae8fd4e2c7a3a9d969e0527102824ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca6c98e77ee8085ea985376866565a60
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0795f47d2679def0fd9d78e38ff80bb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca6c98e77ee8085ea985376866565a60
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()


class PrimitiveOp_02abfb0d9fb4849662e39a8ae8dec8fa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.unfold(input_0, [7, 7], [2, 2], [3, 3, 3, 3], [1, 1])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b896b4d06d5f5edfc48b4579f07a5f27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02abfb0d9fb4849662e39a8ae8dec8fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e3657629e83ae65d99c1227f8c2c4d15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca6c98e77ee8085ea985376866565a60
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c578c3558c257ec6f3a13cf0756d5740(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02abfb0d9fb4849662e39a8ae8dec8fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7c2c2a0b9df7f3b739db26adf295c85f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca6c98e77ee8085ea985376866565a60
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2e1315096f3d9b68b2406cd220e391f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02abfb0d9fb4849662e39a8ae8dec8fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ad1947adc2e6e08ac5b2e74d5ee2d72c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca6c98e77ee8085ea985376866565a60
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_73eed86fc583856f2bb0b688dc6bb3e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca6c98e77ee8085ea985376866565a60
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5fef9fd252faad043b266a17eacd62d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02abfb0d9fb4849662e39a8ae8dec8fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_61429c63179ff861c2ff316ff02c4e90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02abfb0d9fb4849662e39a8ae8dec8fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c20e5a06c5a80650c6db07314b043028(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b6593f195462e9032f7f662c64034a0
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 112, 112], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f20f2cd4353638cd5e046bd87b47929c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca6c98e77ee8085ea985376866565a60
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_97394c2aae15e2d777592dade4f15dcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02abfb0d9fb4849662e39a8ae8dec8fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b1acdd3be8fdcc6b0610643ac6ab7ae2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b6593f195462e9032f7f662c64034a0
    def get_inputs(self):
        return [
            paddle.uniform([10, 32, 112, 112], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_11a0c78a823a4e095f3c23474d9773fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca6c98e77ee8085ea985376866565a60
    def get_inputs(self):
        return [
            paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dae8fd4e2c7a3a9d969e0527102824ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca6c98e77ee8085ea985376866565a60
    def get_inputs(self):
        return [
            paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0795f47d2679def0fd9d78e38ff80bb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca6c98e77ee8085ea985376866565a60
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b896b4d06d5f5edfc48b4579f07a5f27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02abfb0d9fb4849662e39a8ae8dec8fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e3657629e83ae65d99c1227f8c2c4d15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca6c98e77ee8085ea985376866565a60
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c578c3558c257ec6f3a13cf0756d5740(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02abfb0d9fb4849662e39a8ae8dec8fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 256, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7c2c2a0b9df7f3b739db26adf295c85f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca6c98e77ee8085ea985376866565a60
    def get_inputs(self):
        return [
            paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2e1315096f3d9b68b2406cd220e391f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02abfb0d9fb4849662e39a8ae8dec8fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ad1947adc2e6e08ac5b2e74d5ee2d72c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca6c98e77ee8085ea985376866565a60
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_73eed86fc583856f2bb0b688dc6bb3e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca6c98e77ee8085ea985376866565a60
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5fef9fd252faad043b266a17eacd62d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02abfb0d9fb4849662e39a8ae8dec8fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 512, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_61429c63179ff861c2ff316ff02c4e90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02abfb0d9fb4849662e39a8ae8dec8fa
    def get_inputs(self):
        return [
            paddle.uniform([10, 128, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c20e5a06c5a80650c6db07314b043028(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b6593f195462e9032f7f662c64034a0
    def get_inputs(self):
        return [
            paddle.uniform([22, 32, 112, 112], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f20f2cd4353638cd5e046bd87b47929c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca6c98e77ee8085ea985376866565a60
    def get_inputs(self):
        return [
            paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_97394c2aae15e2d777592dade4f15dcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02abfb0d9fb4849662e39a8ae8dec8fa
    def get_inputs(self):
        return [
            paddle.uniform([22, 256, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code == (128 - 134):
                # program paniced.
                raise RuntimeError(f"file {__file__} panicked. stderr: {try_run_stderr}")
        return self._test_entry()



if __name__ == '__main__':
    unittest.main()