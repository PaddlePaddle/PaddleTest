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
class PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2, 3]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_107f56c6e4d588913f4a1fd34154fbc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_61c115bdafceb29e1d9db9a2c1079820(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([1, 92, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_7d7542c39d37839ed6ff48e6279daa57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([22, 2048, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_59a76a3e32e1dd6d998b09d1c8a1eca5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ff529edc6a7ea3fe9579e2bc2f9408e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ffaccecb367766210ca01da857167347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([10, 336, 1, 1], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_6aafc5a747379512c484bba80354f021(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, None, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_207980ec4ee3bde2c898e4ae7690b247(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aafc5a747379512c484bba80354f021
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 3549, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a1af11ec433e8832fd4bc0beb861337a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([10, 60, 1, 1], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_72809f65a83da7c5cc4c6142f1012d8b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1f9752aaa33cb7405acfa059cb02633c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72809f65a83da7c5cc4c6142f1012d8b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3800, 1], dtype='int64'),
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
class TestPrimitiveOp_5a9ddc3b8d8a3d155786782974c5dc3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72809f65a83da7c5cc4c6142f1012d8b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[150, 1], dtype='int64'),
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
class TestPrimitiveOp_9508cbad018d159fa0713eb4455bbad4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([145, 336, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9508cbad018d159fa0713eb4455bbad4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([145, 336, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f8d259641f2f72e31254975e65602542(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72809f65a83da7c5cc4c6142f1012d8b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[40, 1], dtype='int64'),
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
class TestPrimitiveOp_1f9752aaa33cb7405acfa059cb02633c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72809f65a83da7c5cc4c6142f1012d8b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3800, 1], dtype='int64'),
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


class PrimitiveOp_b89a3e834947e5d74540672a672f4a52(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2100cf83fcb88f6c869b957c9560c925(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b89a3e834947e5d74540672a672f4a52
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.1909523010253906], [1.9268392324447632], [1.9188376665115356], [1.9172234535217285], [1.8514010906219482], [2.063127040863037], [2.1504814624786377], [2.2262377738952637], [2.0782384872436523], [1.9154802560806274], [2.066532850265503], [2.037445068359375], [2.0835108757019043], [2.0910696983337402], [2.204735040664673], [2.2618346214294434]], dtype='float32').reshape([16, 1]),
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
class TestPrimitiveOp_a2bddec956eb312eee2797c308bcd81e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b89a3e834947e5d74540672a672f4a52
    def get_inputs(self):
        return [
            paddle.to_tensor([[1.8630328178405762], [2.241473913192749], [1.88625967502594], [2.069567918777466], [2.148874521255493], [2.1069064140319824], [2.201632022857666], [2.1933751106262207], [1.9832137823104858], [2.154857635498047], [1.993782639503479], [1.940382719039917], [2.293872117996216], [2.3069310188293457], [2.1571273803710938], [2.0040831565856934]], dtype='float32').reshape([16, 1]),
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
class TestPrimitiveOp_efef2a412b068ef1c71bfa4eb0604e5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([145, 240, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d00b44b3ccf063c2a1831dc7b7a5b0cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aafc5a747379512c484bba80354f021
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 7581, 4], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_c5398e9a7908ba8eb7dd6f27a1d8e339(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d91bd06b04f2a88bad189ce1946e2b3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5398e9a7908ba8eb7dd6f27a1d8e339
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 18, 64, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d91bd06b04f2a88bad189ce1946e2b3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5398e9a7908ba8eb7dd6f27a1d8e339
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 18, 64, 128], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_e7228768a28240c48dcdd698f2612fa7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, None, 1, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9a9f8ad2079f824af9e7c814715aeabb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7228768a28240c48dcdd698f2612fa7
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 66, 130], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_fc6df4cba07ff8dc2b66e338a0c11bc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aafc5a747379512c484bba80354f021
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4725, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1af7d6c3217c4dee9c45d1f073f3d793(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([22, 60, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_996a9fbd445693fb99453b9b4df7aabd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([1, 872, 1, 1], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5ec0cd58b91a0002a92dcb66e78655e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366
    def get_inputs(self):
        return [
            paddle.uniform([1755, 4, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5ec0cd58b91a0002a92dcb66e78655e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366
    def get_inputs(self):
        return [
            paddle.uniform([1755, 4, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_cd9492730ef8ae2ddd7405177c2a508f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aafc5a747379512c484bba80354f021
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8400, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5b1e67b761e7717ef4aaaf584c81914a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([171, 336, 1, 1], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_a8aaada5dff752c6b8b57ebc3c897074(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 768, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f45b3f4fd9c62cfc265292d31dde6660(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8aaada5dff752c6b8b57ebc3c897074
    def get_inputs(self):
        return [
            paddle.uniform([43, 768, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_207980ec4ee3bde2c898e4ae7690b247(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aafc5a747379512c484bba80354f021
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 3549, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d0f03a988b1ae9571ee63d920a0f2b65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([10, 240, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c554aaf16e534f2596830f37264f087e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366
    def get_inputs(self):
        return [
            paddle.uniform([5589, 4, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c554aaf16e534f2596830f37264f087e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366
    def get_inputs(self):
        return [
            paddle.uniform([5589, 4, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c5994391d91b40b553a10c21572c7531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b89a3e834947e5d74540672a672f4a52
    def get_inputs(self):
        return [
            paddle.uniform([36, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c5994391d91b40b553a10c21572c7531(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b89a3e834947e5d74540672a672f4a52
    def get_inputs(self):
        return [
            paddle.uniform([36, 1], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_6d423954ef9ef462781c241f8e608dae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2, 3]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1000, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2137bc7d0371b20113034951b1d57ff9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d423954ef9ef462781c241f8e608dae
    def get_inputs(self):
        return [
            paddle.uniform([43, 1000, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ffaccecb367766210ca01da857167347(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([10, 336, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9db240e678207c7a1d3e7446d6eb13dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72809f65a83da7c5cc4c6142f1012d8b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[15200, 1], dtype='int64'),
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
class TestPrimitiveOp_9db240e678207c7a1d3e7446d6eb13dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72809f65a83da7c5cc4c6142f1012d8b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[15200, 1], dtype='int64'),
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
class TestPrimitiveOp_11f01c6874025ec359a766989e8ad984(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([10, 36, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_e915bda4de01e4b1c4ce6036b52d30c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([43, 1280, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_aeb37568bf3b6046c996268b05d9e5e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d423954ef9ef462781c241f8e608dae
    def get_inputs(self):
        return [
            paddle.uniform([10, 1000, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8455597eb1c5b79d8c04dd2c54377a29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([10, 480, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_fed7bf48ce6ccf25518b0c6f63c6515c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366
    def get_inputs(self):
        return [
            paddle.uniform([1790, 4, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_fed7bf48ce6ccf25518b0c6f63c6515c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366
    def get_inputs(self):
        return [
            paddle.uniform([1790, 4, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d1978b3a6b3d69dcf09159166c99e5b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([22, 336, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_591540425fd1bcaaa24cc68f10568b6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aafc5a747379512c484bba80354f021
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4116, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_519ed4778a5c6c940f245dc18643ec38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([171, 240, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5b1e67b761e7717ef4aaaf584c81914a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([171, 336, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_40ad384b7c96da92a7b45ffa22c89e6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([22, 1536, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_85885aae50608b5bbb707db61234f2d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b89a3e834947e5d74540672a672f4a52
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.021954298019409], [2.0357892513275146], [1.8439505100250244], [1.8384853601455688], [2.0474965572357178], [2.3225255012512207], [2.2286674976348877], [2.1248745918273926], [2.0436902046203613], [2.229543685913086], [2.0362138748168945], [2.1180944442749023], [2.254608154296875], [1.9478822946548462], [1.9867286682128906], [2.1380062103271484], [2.207724094390869], [2.0267772674560547], [1.8825595378875732], [1.978208065032959], [2.1397013664245605], [2.1785337924957275], [2.1380367279052734], [2.0689496994018555]], dtype='float32').reshape([24, 1]),
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
class TestPrimitiveOp_1b188c06665f98328c0f8190778b1032(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b89a3e834947e5d74540672a672f4a52
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.0444107055664062], [2.048163890838623], [2.1612942218780518], [2.0220460891723633], [2.023146152496338], [1.9601848125457764], [2.1312670707702637], [2.174978256225586], [2.2121052742004395], [2.2835960388183594], [2.0821688175201416], [1.9250227212905884], [1.9156339168548584], [2.0176265239715576], [2.181218385696411], [1.8523257970809937], [1.9184234142303467], [1.973307728767395], [2.0993099212646484], [2.0601232051849365], [1.8775489330291748], [2.2882165908813477], [1.9877315759658813], [2.0468063354492188]], dtype='float32').reshape([24, 1]),
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
class TestPrimitiveOp_aa2901dfef3bc1fb52ad018ae5c0ab90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([171, 60, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a4a47bc9b4a2c6cc69b03e1f8ebd764e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aafc5a747379512c484bba80354f021
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 6069, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_658e1191a86c11eb8fae8cc720045a02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366
    def get_inputs(self):
        return [
            paddle.uniform([1512, 4, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_658e1191a86c11eb8fae8cc720045a02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366
    def get_inputs(self):
        return [
            paddle.uniform([1512, 4, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c5a62c1896bc14469121c3b4a2be5df3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([22, 240, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a08a20491adbebf6656d910ec231f763(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([10, 1536, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_0af464d3bfc027432bf4a3e744f611a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b89a3e834947e5d74540672a672f4a52
    def get_inputs(self):
        return [
            paddle.to_tensor([[1.9469883441925049], [1.9976061582565308], [2.301198720932007], [1.8381811380386353]], dtype='float32').reshape([4, 1]),
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
class TestPrimitiveOp_b0c1041c49d367c1fbf02d443a994c4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b89a3e834947e5d74540672a672f4a52
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.0053937435150146], [1.9138541221618652], [2.0178191661834717], [2.173750400543213]], dtype='float32').reshape([4, 1]),
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
class TestPrimitiveOp_ee870e33f925f598b57d475f69bb3ea3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7228768a28240c48dcdd698f2612fa7
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 70, 134], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_7e8f1da49d07aa940054186243abf17b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_11b65baf5654885b138d109e789808aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e8f1da49d07aa940054186243abf17b
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 104, 101], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a76fc3bb1880dca9f04edda5935e4eac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72809f65a83da7c5cc4c6142f1012d8b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2204, 1], dtype='int64'),
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
class TestPrimitiveOp_792be3c32af524d948825149e84d1eb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([22, 36, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_47366b3a76f4ad0abfd56156227ff637(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7228768a28240c48dcdd698f2612fa7
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 68, 132], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_644d048fedb52b8dc54e371764e0c305(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d423954ef9ef462781c241f8e608dae
    def get_inputs(self):
        return [
            paddle.uniform([11, 1000, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_60d5e4c02866ec265b860fa66b7324d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([145, 60, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8b99135569539a39047579a6b4daca9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5398e9a7908ba8eb7dd6f27a1d8e339
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 36, 32, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8b99135569539a39047579a6b4daca9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5398e9a7908ba8eb7dd6f27a1d8e339
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 36, 32, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_200875c625be6843aa3ed9aea59c5d94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72809f65a83da7c5cc4c6142f1012d8b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[70, 1], dtype='int64'),
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
class TestPrimitiveOp_46ebfcfb0f4785441f2cee73b994bf68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4624f6702b19bcf454e672fa7389f390(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72809f65a83da7c5cc4c6142f1012d8b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[551, 1], dtype='int64'),
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
class TestPrimitiveOp_d1978b3a6b3d69dcf09159166c99e5b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([22, 336, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_95295434e59fc04ffe76f1d21fe4e9ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72809f65a83da7c5cc4c6142f1012d8b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[247, 1], dtype='int64'),
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
class TestPrimitiveOp_d84af74c6e832602dbd968de02bf010b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([10, 2048, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_751111bdcab58c5d8717c6aa1a34383b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72809f65a83da7c5cc4c6142f1012d8b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[950, 1], dtype='int64'),
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
class TestPrimitiveOp_294126b6fe4d60f0911d7f4668cc0419(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366
    def get_inputs(self):
        return [
            paddle.uniform([2007, 4, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_294126b6fe4d60f0911d7f4668cc0419(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366
    def get_inputs(self):
        return [
            paddle.uniform([2007, 4, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_00260ea2287c4ec60ecc3e403d75be9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72809f65a83da7c5cc4c6142f1012d8b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[8816, 1], dtype='int64'),
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
class TestPrimitiveOp_62c5121d9f6817394befc64ab7ee1771(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366
    def get_inputs(self):
        return [
            paddle.uniform([4685, 4, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_62c5121d9f6817394befc64ab7ee1771(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366
    def get_inputs(self):
        return [
            paddle.uniform([4685, 4, 1], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_8f71a431d70e2bc9c70bc159dfd104c6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_11cb274ef26705fa1c92fbf732ad4829(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f71a431d70e2bc9c70bc159dfd104c6
    def get_inputs(self):
        return [
            paddle.uniform([10, 96, 1, 40], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_df1a6fd9e3f49838c7a28a2d6f315c92(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ec7b1d0927c9697c50e424daf6586d2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df1a6fd9e3f49838c7a28a2d6f315c92
    def get_inputs(self):
        return [
            paddle.uniform([1, 2434, 1], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_3a35f7cc8252f59575db121d4aa4e70f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c9f80c945d81967929efaa7377d94b70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a35f7cc8252f59575db121d4aa4e70f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 2434, 1], dtype='int64'),
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
class TestPrimitiveOp_931ee6cba97a9d9d79cc542c24717d10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366
    def get_inputs(self):
        return [
            paddle.uniform([1050, 4, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_931ee6cba97a9d9d79cc542c24717d10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366
    def get_inputs(self):
        return [
            paddle.uniform([1050, 4, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1fc75feb0c2f6c311206ded2fec25855(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aafc5a747379512c484bba80354f021
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 9261, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_29965dae7c570a80fcb898d706dc6a73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a8aaada5dff752c6b8b57ebc3c897074
    def get_inputs(self):
        return [
            paddle.uniform([11, 768, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_acc23dbfb8a3e42e03fdec93d82d8d29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7228768a28240c48dcdd698f2612fa7
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 64, 128], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_3f7612df3520fb9d9ff1ce8d67df9b4a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1000, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f2ff7125298d7c0f9a708e33a21bf9dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f7612df3520fb9d9ff1ce8d67df9b4a
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000, 1, 1], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_52172819fffb2d5d2b985615540c6912(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1000, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fe5fa9f9b4f87e5a52d35eac276ec45f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_52172819fffb2d5d2b985615540c6912
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1885d8a6913e65714629db7b1917ca50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aafc5a747379512c484bba80354f021
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2100, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_afb2f1ca8cb23c55e6a0ef2ce1c471cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([1, 1248, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_127155040aa793f938a3855381597a81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([171, 480, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d4b018c8677cfa041d90c85fc3749afd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([145, 36, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_0da873a450d8e29049b4e594309cf59a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5398e9a7908ba8eb7dd6f27a1d8e339
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 9, 128, 256], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_0da873a450d8e29049b4e594309cf59a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5398e9a7908ba8eb7dd6f27a1d8e339
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 9, 128, 256], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d5c8b79d81d15b0be59989dd65ca315a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366
    def get_inputs(self):
        return [
            paddle.uniform([2394, 4, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d5c8b79d81d15b0be59989dd65ca315a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366
    def get_inputs(self):
        return [
            paddle.uniform([2394, 4, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_7ecfe1a068a12db215ef5705f52ee5b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5398e9a7908ba8eb7dd6f27a1d8e339
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 96, 32, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_7ecfe1a068a12db215ef5705f52ee5b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5398e9a7908ba8eb7dd6f27a1d8e339
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 96, 32, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d0130ddf44d445ac6de94f5d14d81877(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366
    def get_inputs(self):
        return [
            paddle.uniform([3063, 4, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d0130ddf44d445ac6de94f5d14d81877(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366
    def get_inputs(self):
        return [
            paddle.uniform([3063, 4, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f934f8705a18704b8696c7db806bafc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366
    def get_inputs(self):
        return [
            paddle.uniform([3758, 4, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f934f8705a18704b8696c7db806bafc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366
    def get_inputs(self):
        return [
            paddle.uniform([3758, 4, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ceecb1588af84e1c1c24325513bd5b8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5398e9a7908ba8eb7dd6f27a1d8e339
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 24, 128, 256], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ceecb1588af84e1c1c24325513bd5b8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5398e9a7908ba8eb7dd6f27a1d8e339
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 24, 128, 256], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ec3b791ecdedbe89ad2d9c2db98313ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([1, 156, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_2cca7fcc9c2907f43cde7e75f37d7fb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5398e9a7908ba8eb7dd6f27a1d8e339
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 48, 64, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_2cca7fcc9c2907f43cde7e75f37d7fb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5398e9a7908ba8eb7dd6f27a1d8e339
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 48, 64, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c8b588a249c7eb562751193488e9a3fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aafc5a747379512c484bba80354f021
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 11109, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_996a9fbd445693fb99453b9b4df7aabd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([1, 872, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_021706673ac70ed43ce1acd0c5db14e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([22, 480, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5e56dc4d81bfc2472d74a12d51709873(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([145, 480, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f2a891e3649a4202cb0e4b361a3e4c92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f71a431d70e2bc9c70bc159dfd104c6
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 1, 25], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_190879faefced1b4aead33e65c7d661c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([171, 36, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6c373055c43fefde454415cd829f10df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_2efba8a461d20ffa56db76db74e9c03e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b89a3e834947e5d74540672a672f4a52
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.1200664043426514], [2.2037816047668457], [2.2674267292022705], [2.035215139389038], [1.9970215559005737], [2.108041763305664], [2.2404134273529053], [2.1551849842071533], [2.3295679092407227], [2.168734073638916], [2.21038818359375], [2.179452896118164], [2.010749578475952], [2.122589111328125], [1.958745002746582], [2.284149408340454], [2.0211985111236572], [1.967766523361206], [2.109471559524536], [2.301105260848999]], dtype='float32').reshape([20, 1]),
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
class TestPrimitiveOp_9f98057fec65d53f1c24e0112db7c562(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b89a3e834947e5d74540672a672f4a52
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.189807891845703], [1.9733121395111084], [2.0805373191833496], [2.355379343032837], [2.1311275959014893], [2.0823404788970947], [2.287081718444824], [2.026214122772217], [1.9933466911315918], [2.019587755203247], [2.017974853515625], [1.9902701377868652], [2.0971810817718506], [2.303508758544922], [2.238131523132324], [2.1144118309020996], [2.088850498199463], [1.9082940816879272], [2.2732808589935303], [2.281491756439209]], dtype='float32').reshape([20, 1]),
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
class TestPrimitiveOp_95295434e59fc04ffe76f1d21fe4e9ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72809f65a83da7c5cc4c6142f1012d8b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[247, 1], dtype='int64'),
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
class TestPrimitiveOp_46ebfcfb0f4785441f2cee73b994bf68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1f9752aaa33cb7405acfa059cb02633c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72809f65a83da7c5cc4c6142f1012d8b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3800, 1], dtype='int64'),
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
class TestPrimitiveOp_e9ede36e37cf011fa1c359a0c18abae2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df1a6fd9e3f49838c7a28a2d6f315c92
    def get_inputs(self):
        return [
            paddle.uniform([1, 8732, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_fe65f5022636e7ff32a69a8aefa7bd31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a35f7cc8252f59575db121d4aa4e70f
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 8732, 1], dtype='int64'),
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
class TestPrimitiveOp_751111bdcab58c5d8717c6aa1a34383b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72809f65a83da7c5cc4c6142f1012d8b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[950, 1], dtype='int64'),
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
class TestPrimitiveOp_4e0e914ed5c3e1df17e917c5ffa52fbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366
    def get_inputs(self):
        return [
            paddle.uniform([2008, 4, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4e0e914ed5c3e1df17e917c5ffa52fbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366
    def get_inputs(self):
        return [
            paddle.uniform([2008, 4, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a8c3e55cf024c69778b356811cb2217c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d423954ef9ef462781c241f8e608dae
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_200875c625be6843aa3ed9aea59c5d94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72809f65a83da7c5cc4c6142f1012d8b
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[70, 1], dtype='int64'),
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
class TestPrimitiveOp_7e24d3dfd6c6d4ab57f2dfde11cb32ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6aafc5a747379512c484bba80354f021
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 3024, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9e4e3978e071e98d9594baa40230ed77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([11, 1280, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_11c5ea59e77fa9551533af39fd6f2b83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366
    def get_inputs(self):
        return [
            paddle.uniform([4295, 4, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_11c5ea59e77fa9551533af39fd6f2b83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_168370a5e4a84ffdddadc9a2c3674366
    def get_inputs(self):
        return [
            paddle.uniform([4295, 4, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_419410e432cb968e0ca2f2d4b6d640ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bf80f2bfabbd11d972589edaf84b125
    def get_inputs(self):
        return [
            paddle.uniform([1, 624, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_e6abf0f7a52967587bf158fab003e900(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f7612df3520fb9d9ff1ce8d67df9b4a
    def get_inputs(self):
        return [
            paddle.uniform([10, 1000, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_200b8ee4767cf6cf1965aece8b01db51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_52172819fffb2d5d2b985615540c6912
    def get_inputs(self):
        return [
            paddle.uniform([10, 1000, 1], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2, 3]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_463980bce97715e9b1efcb56942ab82a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_3fdec699ae8620f465f251a1d9ad5f77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([1, 92, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5335da310ab758484d6cb970338ee225(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([22, 2048, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_02e14fd0ee3c257a7a5f1118372cef95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ae8e12185159bfd8ce466a1f9130e761(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9f6d879e8e8806d8c9dd748eea3c8f00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([10, 336, 1, 1], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_cb41464438e84d3938aae4495fd4e2af(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_56f93cf119ae4079622ac343e942ddc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb41464438e84d3938aae4495fd4e2af
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 3549, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5ee17f0e727de660ef580b350eccdc67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([10, 60, 1, 1], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_f26c7d7e78b97ea05a7af11431b7dc29(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eb94ebc53d62fa11f4a3a51a16ac1e29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f26c7d7e78b97ea05a7af11431b7dc29
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3800, 1], dtype='int64'),
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
class TestPrimitiveOp_0fd000a881d3eebe96b617257191336c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f26c7d7e78b97ea05a7af11431b7dc29
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[150, 1], dtype='int64'),
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
class TestPrimitiveOp_f7d9f09f9d801fea46432cbb1eec82e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([145, 336, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f7d9f09f9d801fea46432cbb1eec82e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([145, 336, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_70adad6afa5c6c4a78d20057bb91ea86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f26c7d7e78b97ea05a7af11431b7dc29
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[40, 1], dtype='int64'),
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
class TestPrimitiveOp_eb94ebc53d62fa11f4a3a51a16ac1e29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f26c7d7e78b97ea05a7af11431b7dc29
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3800, 1], dtype='int64'),
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


class PrimitiveOp_c8abc660aa844735df7f79f4152a427d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1b36c1fbf75fe1dfb9038c00eb4683aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8abc660aa844735df7f79f4152a427d
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.1909523010253906], [1.9268392324447632], [1.9188376665115356], [1.9172234535217285], [1.8514010906219482], [2.063127040863037], [2.1504814624786377], [2.2262377738952637], [2.0782384872436523], [1.9154802560806274], [2.066532850265503], [2.037445068359375], [2.0835108757019043], [2.0910696983337402], [2.204735040664673], [2.2618346214294434]], dtype='float32').reshape([16, 1]),
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
class TestPrimitiveOp_b65a996094d7cef1546eaff5e62ba1ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8abc660aa844735df7f79f4152a427d
    def get_inputs(self):
        return [
            paddle.to_tensor([[1.8630328178405762], [2.241473913192749], [1.88625967502594], [2.069567918777466], [2.148874521255493], [2.1069064140319824], [2.201632022857666], [2.1933751106262207], [1.9832137823104858], [2.154857635498047], [1.993782639503479], [1.940382719039917], [2.293872117996216], [2.3069310188293457], [2.1571273803710938], [2.0040831565856934]], dtype='float32').reshape([16, 1]),
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
class TestPrimitiveOp_ca7675f56b1e80850c670e8f5e396c97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([145, 240, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5f3fef195c76a7041da7ad8baacbbecb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb41464438e84d3938aae4495fd4e2af
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 7581, 4], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_883b4062ceb73368385ec3ca83d65e32(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8732d397cd6083a774d13656a4e385ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_883b4062ceb73368385ec3ca83d65e32
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 18, 64, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_8732d397cd6083a774d13656a4e385ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_883b4062ceb73368385ec3ca83d65e32
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 18, 64, 128], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_e4a763da5fbe4f62f7e8ed4cd72de976(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2389daa67bfd6ba953cd3a5d488a65b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4a763da5fbe4f62f7e8ed4cd72de976
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 66, 130], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a98c4471ab28e2277fb4e137b92eecf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb41464438e84d3938aae4495fd4e2af
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4725, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b4f71b328b3726b0dc46b0aff14fdf0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([22, 60, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_47c09ba1238a9f0651d15e1198e76696(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([1, 872, 1, 1], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7438ae50b1049a0360751a4f37bb23db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([1755, 4, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_7438ae50b1049a0360751a4f37bb23db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([1755, 4, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6f1983aaf0f2f2cc45f4d55aa2021997(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb41464438e84d3938aae4495fd4e2af
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8400, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9c0409136730c7d58ef06e176490e43d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([171, 336, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b597523a04a1961c3a48e94336450c5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f71a431d70e2bc9c70bc159dfd104c6
    def get_inputs(self):
        return [
            paddle.uniform([43, 768, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_56f93cf119ae4079622ac343e942ddc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb41464438e84d3938aae4495fd4e2af
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 3549, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_511ebfde8c6216a4a237a1836c69ac8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([10, 240, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_54eb58dce03334568c3fd4f0c62069ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([5589, 4, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_54eb58dce03334568c3fd4f0c62069ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([5589, 4, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_3bdf8e51cc45fb0769d08d39d55ac351(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8abc660aa844735df7f79f4152a427d
    def get_inputs(self):
        return [
            paddle.uniform([36, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_3bdf8e51cc45fb0769d08d39d55ac351(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8abc660aa844735df7f79f4152a427d
    def get_inputs(self):
        return [
            paddle.uniform([36, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ee3fe9864a798604907bf8f2ee66a796(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([43, 1000, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9f6d879e8e8806d8c9dd748eea3c8f00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([10, 336, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b394fc6ca83f795ba5c89c947e780fc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f26c7d7e78b97ea05a7af11431b7dc29
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[15200, 1], dtype='int64'),
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
class TestPrimitiveOp_b394fc6ca83f795ba5c89c947e780fc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f26c7d7e78b97ea05a7af11431b7dc29
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[15200, 1], dtype='int64'),
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
class TestPrimitiveOp_4953b8c6f122350a8debd015d1bfd97e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([10, 36, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6cdd0fc4711249b0b38d895ae18dfaad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([43, 1280, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_ddeab4cd6a9d3c90860ae0533ba1ccbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([10, 1000, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f5fb4ae00e96e60b7e57aa2f857525aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([10, 480, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_0fb178eed31ceae710aee92f08ae42ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([1790, 4, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_0fb178eed31ceae710aee92f08ae42ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([1790, 4, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_954bfac9bc21a333f24b06008f6dd186(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([22, 336, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5facc11e8b6ed63bff5abded60dfd679(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb41464438e84d3938aae4495fd4e2af
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4116, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_93a01b6c0a984b7df39dd2f4729e5b7e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([171, 240, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_9c0409136730c7d58ef06e176490e43d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([171, 336, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c0f874a29a642d876e8156db7d4f3fde(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([22, 1536, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_59226001d436b8bb2a75d0d730e04cc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8abc660aa844735df7f79f4152a427d
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.021954298019409], [2.0357892513275146], [1.8439505100250244], [1.8384853601455688], [2.0474965572357178], [2.3225255012512207], [2.2286674976348877], [2.1248745918273926], [2.0436902046203613], [2.229543685913086], [2.0362138748168945], [2.1180944442749023], [2.254608154296875], [1.9478822946548462], [1.9867286682128906], [2.1380062103271484], [2.207724094390869], [2.0267772674560547], [1.8825595378875732], [1.978208065032959], [2.1397013664245605], [2.1785337924957275], [2.1380367279052734], [2.0689496994018555]], dtype='float32').reshape([24, 1]),
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
class TestPrimitiveOp_34858e641f0af99eb0af127a2a6aeb72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8abc660aa844735df7f79f4152a427d
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.0444107055664062], [2.048163890838623], [2.1612942218780518], [2.0220460891723633], [2.023146152496338], [1.9601848125457764], [2.1312670707702637], [2.174978256225586], [2.2121052742004395], [2.2835960388183594], [2.0821688175201416], [1.9250227212905884], [1.9156339168548584], [2.0176265239715576], [2.181218385696411], [1.8523257970809937], [1.9184234142303467], [1.973307728767395], [2.0993099212646484], [2.0601232051849365], [1.8775489330291748], [2.2882165908813477], [1.9877315759658813], [2.0468063354492188]], dtype='float32').reshape([24, 1]),
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
class TestPrimitiveOp_0296e20b4c66637eba3a9dd27bbd623d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([171, 60, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_90d9611a6eb0dff9d3eb213614abf901(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb41464438e84d3938aae4495fd4e2af
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 6069, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_cb47b4dc0a4b3103e06b717d5b672fca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([1512, 4, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_cb47b4dc0a4b3103e06b717d5b672fca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([1512, 4, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d520e3988bcd6ce25132d8f0d7a3c44c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([22, 240, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_2f1610b4fa26915daf6a11b5c26e1ebf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([10, 1536, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_74a1f2575bf7d991bb8a9446e4611c6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8abc660aa844735df7f79f4152a427d
    def get_inputs(self):
        return [
            paddle.to_tensor([[1.9469883441925049], [1.9976061582565308], [2.301198720932007], [1.8381811380386353]], dtype='float32').reshape([4, 1]),
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
class TestPrimitiveOp_21b1421dbee985533480c0f5a81cb73e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8abc660aa844735df7f79f4152a427d
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.0053937435150146], [1.9138541221618652], [2.0178191661834717], [2.173750400543213]], dtype='float32').reshape([4, 1]),
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
class TestPrimitiveOp_33710390fee8418e9de45b75cc929078(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4a763da5fbe4f62f7e8ed4cd72de976
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 70, 134], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_cc02d45b36a1ddbaa8c3fc23bdb72578(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4a763da5fbe4f62f7e8ed4cd72de976
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 104, 101], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_dbc2d4453a8f0253f1c78d7e6a51d849(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f26c7d7e78b97ea05a7af11431b7dc29
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[2204, 1], dtype='int64'),
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
class TestPrimitiveOp_a8bb231543ea167c59e4636179b06670(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([22, 36, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1b23cacef947e56f7cc143472ab834b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4a763da5fbe4f62f7e8ed4cd72de976
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 68, 132], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_fa3e4543a0273d5303cde003350cb39e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([11, 1000, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_76d58ec9686d2b07fe302903a22e3c4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([145, 60, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_e998c0a1a3b4d146cfd27c1aab28368f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_883b4062ceb73368385ec3ca83d65e32
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 36, 32, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_e998c0a1a3b4d146cfd27c1aab28368f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_883b4062ceb73368385ec3ca83d65e32
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 36, 32, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_7d8081fcd6c0f9e5e7792d394a329f4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f26c7d7e78b97ea05a7af11431b7dc29
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[70, 1], dtype='int64'),
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
class TestPrimitiveOp_3dd28cd2fac5aac9bff1565954279c92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_1c418de12de37eea5d8dbd1394791760(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f26c7d7e78b97ea05a7af11431b7dc29
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[551, 1], dtype='int64'),
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
class TestPrimitiveOp_954bfac9bc21a333f24b06008f6dd186(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([22, 336, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_491cadfe7acb454e7cd9a7bb65c08f95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f26c7d7e78b97ea05a7af11431b7dc29
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[247, 1], dtype='int64'),
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
class TestPrimitiveOp_a5ddbbf51553b7ad1500b465570d8938(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([10, 2048, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4fa455f72ac56584f626f17ec2f0b77c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f26c7d7e78b97ea05a7af11431b7dc29
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[950, 1], dtype='int64'),
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
class TestPrimitiveOp_e00cb44feecc0e26890450878c024d2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([2007, 4, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_e00cb44feecc0e26890450878c024d2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([2007, 4, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_78b1efc28aebfe35fdc80ec2d3e37e4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f26c7d7e78b97ea05a7af11431b7dc29
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[8816, 1], dtype='int64'),
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
class TestPrimitiveOp_2bf8b1bb5f1f37ebd0541db906bae4eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([4685, 4, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_2bf8b1bb5f1f37ebd0541db906bae4eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([4685, 4, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_11cb274ef26705fa1c92fbf732ad4829(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f71a431d70e2bc9c70bc159dfd104c6
    def get_inputs(self):
        return [
            paddle.uniform([10, 96, 1, 40], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b6c0bbceb138ef8a128e5c458a5bee1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 2434, 1], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_8428a6729f99ba67bba319de58dc7901(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f7de54a09fbb6b6389d89d325eb369ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8428a6729f99ba67bba319de58dc7901
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 2434, 1], dtype='int64'),
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
class TestPrimitiveOp_d61550105583aca1688951a880843ffa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([1050, 4, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d61550105583aca1688951a880843ffa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([1050, 4, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5220228431db5661fdcf9489e23f6748(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb41464438e84d3938aae4495fd4e2af
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 9261, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_02cde49b67196e2951a59991b2262a7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f71a431d70e2bc9c70bc159dfd104c6
    def get_inputs(self):
        return [
            paddle.uniform([11, 768, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_798b33e970b7441fefc93a2a13e897c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e4a763da5fbe4f62f7e8ed4cd72de976
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 64, 128], dtype='float32', min=0, max=0.5),
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


class PrimitiveOp_9c3f13317024517a9068de8594946faa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        return paddle._C_ops.squeeze(input_0, input_1), None

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_213669359003cc149d24366a85686806(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c3f13317024517a9068de8594946faa
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_e72659ad328a84cbf860c60be4515145(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_7f4057b67fc4dffbb5025c54a474ded6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb41464438e84d3938aae4495fd4e2af
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2100, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_bf6de3c5eac246360880f3dcbc1074d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1248, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_92073823f2b706fec35eb23e4ed50d46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([171, 480, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_4422b810c025519c8ff9d20276ddcd9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([145, 36, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a6443a21023a1c981b5b35297f8b6a8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_883b4062ceb73368385ec3ca83d65e32
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 9, 128, 256], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_a6443a21023a1c981b5b35297f8b6a8c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_883b4062ceb73368385ec3ca83d65e32
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 9, 128, 256], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_406861692ca05dfcd2aed184c261878f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([2394, 4, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_406861692ca05dfcd2aed184c261878f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([2394, 4, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f0d0f07fcad0f680e19fba854c526c18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_883b4062ceb73368385ec3ca83d65e32
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 96, 32, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f0d0f07fcad0f680e19fba854c526c18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_883b4062ceb73368385ec3ca83d65e32
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 96, 32, 64], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c3c004b8d4928826ebf2269fa5dae835(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([3063, 4, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_c3c004b8d4928826ebf2269fa5dae835(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([3063, 4, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_2314f55c50c4f60511ac93c1d6aa805d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([3758, 4, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_2314f55c50c4f60511ac93c1d6aa805d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([3758, 4, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_3396588d309b55289228e571f89558e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_883b4062ceb73368385ec3ca83d65e32
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 24, 128, 256], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_3396588d309b55289228e571f89558e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_883b4062ceb73368385ec3ca83d65e32
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 24, 128, 256], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_6624b86c7fd1dc07b0e47aa01bff95ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([1, 156, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b29eabf4987eff07da81a477052869e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_883b4062ceb73368385ec3ca83d65e32
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 48, 64, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_b29eabf4987eff07da81a477052869e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_883b4062ceb73368385ec3ca83d65e32
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 48, 64, 128], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_13abaf06f916901d1b041dd5a4458aac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb41464438e84d3938aae4495fd4e2af
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 11109, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_47c09ba1238a9f0651d15e1198e76696(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([1, 872, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_fe67c8f5a6d68f16068826e9656d92fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([22, 480, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_d92769d2f1bf3b42588b87e1b85af57d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([145, 480, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_f2a891e3649a4202cb0e4b361a3e4c92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f71a431d70e2bc9c70bc159dfd104c6
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 1, 25], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_367324e02664a301ad4d638057893f0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([171, 36, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_879cc6cdf15daa6f97276f93db0fec57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_efc2ddf18802ce7c0acd60efb9db20ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8abc660aa844735df7f79f4152a427d
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.1200664043426514], [2.2037816047668457], [2.2674267292022705], [2.035215139389038], [1.9970215559005737], [2.108041763305664], [2.2404134273529053], [2.1551849842071533], [2.3295679092407227], [2.168734073638916], [2.21038818359375], [2.179452896118164], [2.010749578475952], [2.122589111328125], [1.958745002746582], [2.284149408340454], [2.0211985111236572], [1.967766523361206], [2.109471559524536], [2.301105260848999]], dtype='float32').reshape([20, 1]),
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
class TestPrimitiveOp_2ae0af8cc62be65ac1f93858e539c6b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8abc660aa844735df7f79f4152a427d
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.189807891845703], [1.9733121395111084], [2.0805373191833496], [2.355379343032837], [2.1311275959014893], [2.0823404788970947], [2.287081718444824], [2.026214122772217], [1.9933466911315918], [2.019587755203247], [2.017974853515625], [1.9902701377868652], [2.0971810817718506], [2.303508758544922], [2.238131523132324], [2.1144118309020996], [2.088850498199463], [1.9082940816879272], [2.2732808589935303], [2.281491756439209]], dtype='float32').reshape([20, 1]),
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
class TestPrimitiveOp_491cadfe7acb454e7cd9a7bb65c08f95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f26c7d7e78b97ea05a7af11431b7dc29
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[247, 1], dtype='int64'),
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
class TestPrimitiveOp_3dd28cd2fac5aac9bff1565954279c92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_eb94ebc53d62fa11f4a3a51a16ac1e29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f26c7d7e78b97ea05a7af11431b7dc29
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[3800, 1], dtype='int64'),
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
class TestPrimitiveOp_6cd5d2d104edc0075ce6e7265b629156(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 8732, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_70ebd1d32ee7cbe80ad751bf1c4675cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8428a6729f99ba67bba319de58dc7901
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[1, 8732, 1], dtype='int64'),
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
class TestPrimitiveOp_4fa455f72ac56584f626f17ec2f0b77c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f26c7d7e78b97ea05a7af11431b7dc29
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[950, 1], dtype='int64'),
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
class TestPrimitiveOp_e83c068a80f21442f90dd117033e9a13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([2008, 4, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_e83c068a80f21442f90dd117033e9a13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([2008, 4, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_123c07f1dad265be03150f516702f87c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_7d8081fcd6c0f9e5e7792d394a329f4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f26c7d7e78b97ea05a7af11431b7dc29
    def get_inputs(self):
        return [
            paddle.randint(low=0, high=3, shape=[70, 1], dtype='int64'),
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
class TestPrimitiveOp_fc870390d6629627571b4503779d2ff4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb41464438e84d3938aae4495fd4e2af
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 3024, 4], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_fd48a84f6960beb8745e7c4fde126ea9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([11, 1280, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_7398a9a5b0b00d293c02804f610530de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([4295, 4, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_7398a9a5b0b00d293c02804f610530de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([4295, 4, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_32daabdcaa62c0c42e09a67246cc1862(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40c80e7b3d0d94ef03edce2e8196739b
    def get_inputs(self):
        return [
            paddle.uniform([1, 624, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_da5f2d715bd1b362beb542b8e8ee79d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c3f13317024517a9068de8594946faa
    def get_inputs(self):
        return [
            paddle.uniform([10, 1000, 1, 1], dtype='float32', min=0, max=0.5),
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
class TestPrimitiveOp_5f58f82994e0bb654ec0dee4a888a3b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efdbd3d5ef099ac9c4c3ff68109229ef
    def get_inputs(self):
        return [
            paddle.uniform([10, 1000, 1], dtype='float32', min=0, max=0.5),
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