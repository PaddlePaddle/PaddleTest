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
class PrimitiveOp_7eed7bd94a8d77db76bcaef980c0a935(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2, 3]
        return (lambda x, f: f(x))(paddle._C_ops.squeeze(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0293603bd87ccff5f2afff81ac553c3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eed7bd94a8d77db76bcaef980c0a935
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_74725c3b20d0128ae5bb612b1ae36f1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eed7bd94a8d77db76bcaef980c0a935
    def get_inputs(self):
        return [
            paddle.uniform([1, 92, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4d88a1a7ae887b39ecb70379eaf1fed6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eed7bd94a8d77db76bcaef980c0a935
    def get_inputs(self):
        return [
            paddle.uniform([22, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0afe5925ff3200be57dd44b0036cd920(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eed7bd94a8d77db76bcaef980c0a935
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_4218d9ce559ae34de7838647c2d97a99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eed7bd94a8d77db76bcaef980c0a935
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_be75db550aed03a01495a49145842cad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eed7bd94a8d77db76bcaef980c0a935
    def get_inputs(self):
        return [
            paddle.uniform([10, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ec2127f176aa380a1f31fdc5a6077dd4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        return (lambda x, f: f(x))(paddle._C_ops.squeeze(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_414f6ea26ce113c56b23413c14872c5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec2127f176aa380a1f31fdc5a6077dd4
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 3549, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bf28d997a8d9f8e3c1c5bca92b8ae490(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eed7bd94a8d77db76bcaef980c0a935
    def get_inputs(self):
        return [
            paddle.uniform([10, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f9fae7fb92fb27ab7f86b90700f1e45b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        return (lambda x, f: f(x))(paddle._C_ops.squeeze(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_116bafbc4a1b333fd00e02935a7092a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9fae7fb92fb27ab7f86b90700f1e45b
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[3800, 1], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0bf808480d6c07de8c6019c1ef60530f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9fae7fb92fb27ab7f86b90700f1e45b
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[150, 1], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6bb41cfac1234fb1b40cc1dfceb9e1f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eed7bd94a8d77db76bcaef980c0a935
    def get_inputs(self):
        return [
            paddle.uniform([145, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2419c3248b7c9931846f329f4612f9bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9fae7fb92fb27ab7f86b90700f1e45b
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[40, 1], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_efd3a8340e42e76dd325bcdaa5b00ff7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        return (lambda x, f: f(x))(paddle._C_ops.squeeze(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_16842df92f4c7b6b53e5f92e289c1a49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efd3a8340e42e76dd325bcdaa5b00ff7
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.0899338722229004], [2.0193772315979004], [2.2738912105560303], [1.987199068069458], [2.100257396697998], [2.1881446838378906], [1.9883549213409424], [1.8823989629745483], [2.172023057937622], [2.0769574642181396], [2.3263778686523438], [1.9662823677062988], [2.183725357055664], [2.3208024501800537], [2.061213493347168], [2.132282018661499]], dtype='float32').reshape([16, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dff4cc92f28429a09b8d7aa4134b0a51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efd3a8340e42e76dd325bcdaa5b00ff7
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.163323163986206], [1.8874764442443848], [1.9917770624160767], [1.9982393980026245], [2.045048475265503], [2.1794745922088623], [1.9257103204727173], [2.0796351432800293], [2.0372986793518066], [2.3143351078033447], [2.018178939819336], [2.2048075199127197], [2.2676215171813965], [2.1206114292144775], [1.9648325443267822], [2.009826421737671]], dtype='float32').reshape([16, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b7a0e04c5a2248bd755c4e94f36d7021(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eed7bd94a8d77db76bcaef980c0a935
    def get_inputs(self):
        return [
            paddle.uniform([145, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cd24656872ff4c343d56210fd5453698(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec2127f176aa380a1f31fdc5a6077dd4
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 7581, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_582935adcab9770c19132d727a72c378(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        return (lambda x, f: f(x))(paddle._C_ops.squeeze(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1e8a9518cef0ff14391be87d32625e72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_582935adcab9770c19132d727a72c378
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 18, 64, 128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d90e56a81efe2292f832d160b61def88(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        return (lambda x, f: f(x))(paddle._C_ops.squeeze(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4b5a0ff2302d9c68dbbb5aae16c77e6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d90e56a81efe2292f832d160b61def88
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 66, 130], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_db286822f0db32a32bd0218f8cfe11f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec2127f176aa380a1f31fdc5a6077dd4
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4725, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_97fe22687775ea4e2d1e43aecf0da074(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eed7bd94a8d77db76bcaef980c0a935
    def get_inputs(self):
        return [
            paddle.uniform([22, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1002d3415a6d3786b8ef99ef9568c0fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eed7bd94a8d77db76bcaef980c0a935
    def get_inputs(self):
        return [
            paddle.uniform([1, 872, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9668bf902e82eb0424ae466859a24798(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        return (lambda x, f: f(x))(paddle._C_ops.squeeze(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dcb158b973c0665eaff9902702ddeae6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9668bf902e82eb0424ae466859a24798
    def get_inputs(self):
        return [
            paddle.uniform([1790, 4, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a3b1075100d1d028dcd6d7ed123b0770(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec2127f176aa380a1f31fdc5a6077dd4
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8400, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a2be76fa9729b2460dfe78bbc566f847(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eed7bd94a8d77db76bcaef980c0a935
    def get_inputs(self):
        return [
            paddle.uniform([171, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5b5e63ddf040121552b163a7c1c7907d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        return (lambda x, f: f(x))(paddle._C_ops.squeeze(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_06a5e8ab0c1a63d73e4c30176353148c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b5e63ddf040121552b163a7c1c7907d
    def get_inputs(self):
        return [
            paddle.uniform([43, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0d12e00f91ce91d9eaca4236e9ff5fdf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eed7bd94a8d77db76bcaef980c0a935
    def get_inputs(self):
        return [
            paddle.uniform([10, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_35b99117c9b9ccd922fc82008dad567d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9668bf902e82eb0424ae466859a24798
    def get_inputs(self):
        return [
            paddle.uniform([5590, 4, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_adf2fbc9f0bd411b80d09e7ee65a1162(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efd3a8340e42e76dd325bcdaa5b00ff7
    def get_inputs(self):
        return [
            paddle.uniform([36, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_da81612c81591441ebdc63c001fbac88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eed7bd94a8d77db76bcaef980c0a935
    def get_inputs(self):
        return [
            paddle.uniform([43, 1000, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ce84b5007427ad74558e506a3e10c691(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9fae7fb92fb27ab7f86b90700f1e45b
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[15200, 1], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4990f121c7f912edac5437dd799d9bd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eed7bd94a8d77db76bcaef980c0a935
    def get_inputs(self):
        return [
            paddle.uniform([10, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dd5d547a4fa33e3b9f2206829eab96f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eed7bd94a8d77db76bcaef980c0a935
    def get_inputs(self):
        return [
            paddle.uniform([43, 1280, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4c7fc01603f2f8320182bac7e48f2ca9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eed7bd94a8d77db76bcaef980c0a935
    def get_inputs(self):
        return [
            paddle.uniform([10, 1000, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f83db4bd3b66a4886c5c2a412523a279(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eed7bd94a8d77db76bcaef980c0a935
    def get_inputs(self):
        return [
            paddle.uniform([10, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_94d1427c93b3d9de9f4039040a62c0d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eed7bd94a8d77db76bcaef980c0a935
    def get_inputs(self):
        return [
            paddle.uniform([22, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_085aef13a64e5a0d39a52469627e1ee0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec2127f176aa380a1f31fdc5a6077dd4
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4116, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0a122472af6c7712d2f391fd368619d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eed7bd94a8d77db76bcaef980c0a935
    def get_inputs(self):
        return [
            paddle.uniform([171, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_984512d5ffe9a4d6ad27835c3994b2e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eed7bd94a8d77db76bcaef980c0a935
    def get_inputs(self):
        return [
            paddle.uniform([22, 1536, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_05d330843e12eff75f5bc79eca297fe7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efd3a8340e42e76dd325bcdaa5b00ff7
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.006956100463867], [1.9117376804351807], [2.017982006072998], [2.205371856689453], [2.211639404296875], [2.1194374561309814], [1.9514344930648804], [1.9654366970062256], [2.209026336669922], [2.0346641540527344], [2.2032999992370605], [1.9927783012390137], [1.9959903955459595], [2.281437873840332], [2.1363065242767334], [1.9054741859436035], [1.847438097000122], [2.261545419692993], [2.121845006942749], [2.2606067657470703], [2.0738754272460938], [2.1228108406066895], [1.9678256511688232], [2.1302120685577393]], dtype='float32').reshape([24, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6d2382bfab34916b7cf29f54727fbe77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efd3a8340e42e76dd325bcdaa5b00ff7
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.0476019382476807], [2.232367992401123], [2.1287949085235596], [1.9709757566452026], [2.2109475135803223], [2.2074756622314453], [1.9347628355026245], [1.9760403633117676], [2.2215323448181152], [1.88779878616333], [1.9031810760498047], [2.0182852745056152], [2.1343891620635986], [1.8733372688293457], [2.20819091796875], [2.1640238761901855], [2.239391326904297], [1.915928602218628], [2.213580846786499], [1.9242842197418213], [2.271833896636963], [1.8983896970748901], [2.3853492736816406], [1.7908287048339844]], dtype='float32').reshape([24, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_895c9c2fb62207f383f6df719860d52f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eed7bd94a8d77db76bcaef980c0a935
    def get_inputs(self):
        return [
            paddle.uniform([171, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f0640ddea3e2a014393fdb35879db5b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec2127f176aa380a1f31fdc5a6077dd4
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 6069, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9735c4190de4211fa37d20d01b284629(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9668bf902e82eb0424ae466859a24798
    def get_inputs(self):
        return [
            paddle.uniform([1532, 4, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4c624b0d779253f6d414701209293241(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eed7bd94a8d77db76bcaef980c0a935
    def get_inputs(self):
        return [
            paddle.uniform([22, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4c2d5c64a6baecaf17d21226c5acdff6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eed7bd94a8d77db76bcaef980c0a935
    def get_inputs(self):
        return [
            paddle.uniform([10, 1536, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_663171d5efd49e063d26ad7311da4b39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efd3a8340e42e76dd325bcdaa5b00ff7
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.241544723510742], [2.0525031089782715], [1.862670660018921], [2.278165578842163]], dtype='float32').reshape([4, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cca7b707ac580b02e89d21bc8c79b59e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efd3a8340e42e76dd325bcdaa5b00ff7
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.1163506507873535], [2.1665420532226562], [2.255371570587158], [1.956946611404419]], dtype='float32').reshape([4, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0b41ad2f917efcea885e1ff9fa623b1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d90e56a81efe2292f832d160b61def88
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 70, 134], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_67e9af0739d17b09efb5e35572c14ae8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d90e56a81efe2292f832d160b61def88
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 104, 101], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9a5be87dd674c68f9e6f22331942694b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9fae7fb92fb27ab7f86b90700f1e45b
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[2204, 1], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_32571cd8fca2fd096900e46ad5c787f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eed7bd94a8d77db76bcaef980c0a935
    def get_inputs(self):
        return [
            paddle.uniform([22, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9d356539a0f33a290a02bdf9368f8668(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d90e56a81efe2292f832d160b61def88
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 68, 132], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f12e248bc12b54255c7cdad880409e17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eed7bd94a8d77db76bcaef980c0a935
    def get_inputs(self):
        return [
            paddle.uniform([11, 1000, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c890d47292e4291b7013a3f85a8e764b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eed7bd94a8d77db76bcaef980c0a935
    def get_inputs(self):
        return [
            paddle.uniform([145, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9ea5bc4728fe7b5f99fbe83f1c79d7f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_582935adcab9770c19132d727a72c378
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 36, 32, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5dab0eb06dc0b535d999d9fe850f5b02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9fae7fb92fb27ab7f86b90700f1e45b
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[70, 1], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0569109834e1c23e693419a1f62096f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eed7bd94a8d77db76bcaef980c0a935
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_6f2ce88087c10563a6270958a4de5920(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9fae7fb92fb27ab7f86b90700f1e45b
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[551, 1], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eae27ea44b8b09a5a4dc827bac57edc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9fae7fb92fb27ab7f86b90700f1e45b
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[247, 1], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_de1563d2513f0765a8247e04645bf67f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eed7bd94a8d77db76bcaef980c0a935
    def get_inputs(self):
        return [
            paddle.uniform([10, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1b967ef252bf282d5e29ac0d42d31967(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9fae7fb92fb27ab7f86b90700f1e45b
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[950, 1], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ea76d75d684ac02860aded3c71405d79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9668bf902e82eb0424ae466859a24798
    def get_inputs(self):
        return [
            paddle.uniform([2029, 4, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_547297125f98449673cf1f420315aabf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9fae7fb92fb27ab7f86b90700f1e45b
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[8816, 1], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_64493d2ad4348559dd0533930c98e6c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9668bf902e82eb0424ae466859a24798
    def get_inputs(self):
        return [
            paddle.uniform([4671, 4, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4acf5c2a3f1075c6758c893c6fddab80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b5e63ddf040121552b163a7c1c7907d
    def get_inputs(self):
        return [
            paddle.uniform([10, 96, 1, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c8ec0e5c773353ddcb4ffd1d8ebf32b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9668bf902e82eb0424ae466859a24798
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

class PrimitiveOp_eec651cb29c8b26e41485f015fcac29a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        return (lambda x, f: f(x))(paddle._C_ops.squeeze(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e079d9afe9a216e2c35183442d1c8a11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eec651cb29c8b26e41485f015fcac29a
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 2434, 1], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_50afb615032fc20860db647ae47f38e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9668bf902e82eb0424ae466859a24798
    def get_inputs(self):
        return [
            paddle.uniform([1040, 4, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_be1b6c88301b61fa77ae768af22791b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec2127f176aa380a1f31fdc5a6077dd4
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 9261, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c70f7788e3a26c36f054b7d174a182e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b5e63ddf040121552b163a7c1c7907d
    def get_inputs(self):
        return [
            paddle.uniform([11, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6cb0a155129935693e029f5bad91aae6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d90e56a81efe2292f832d160b61def88
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 64, 128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c7390a74ffd923c3cd1fbe59a6c5eed8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        return (lambda x, f: f(x))(paddle._C_ops.squeeze(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6635ac9c4e88713aa5e292bdce5c3fda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7390a74ffd923c3cd1fbe59a6c5eed8
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7f90bddae4349bf78c8e024364711a6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9668bf902e82eb0424ae466859a24798
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f5dcf75871e9d4af189a5a4185bf85b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec2127f176aa380a1f31fdc5a6077dd4
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2100, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_42a1fad5ff647eb6c5868eeb68906d75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eed7bd94a8d77db76bcaef980c0a935
    def get_inputs(self):
        return [
            paddle.uniform([1, 1248, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_95a30bb17c42d018935dff3e7f34938f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eed7bd94a8d77db76bcaef980c0a935
    def get_inputs(self):
        return [
            paddle.uniform([171, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5e35478bdef270e45f86c02812375901(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eed7bd94a8d77db76bcaef980c0a935
    def get_inputs(self):
        return [
            paddle.uniform([145, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ef0b9356808737f0365a287187c22ea7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_582935adcab9770c19132d727a72c378
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 9, 128, 256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f6afaf5cb4939bb169ef75508e71be0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9668bf902e82eb0424ae466859a24798
    def get_inputs(self):
        return [
            paddle.uniform([2361, 4, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e231e13c0f924d3502afee51a3bf0ddc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_582935adcab9770c19132d727a72c378
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 96, 32, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_952395b1be0c07c59a3f7db2cd4df603(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9668bf902e82eb0424ae466859a24798
    def get_inputs(self):
        return [
            paddle.uniform([3043, 4, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8a33945c941af6816f55c88a4fa2c7da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9668bf902e82eb0424ae466859a24798
    def get_inputs(self):
        return [
            paddle.uniform([3752, 4, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_39dae53bfdcf62cdbcbcdfc71b884091(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_582935adcab9770c19132d727a72c378
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 24, 128, 256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_52b613ada57f96a89f1d0898e1b0a720(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eed7bd94a8d77db76bcaef980c0a935
    def get_inputs(self):
        return [
            paddle.uniform([1, 156, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f43b187f56d45b0a014ba848a60baf68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_582935adcab9770c19132d727a72c378
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 48, 64, 128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1911dc5c7420c44df782ef205c6e9e75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec2127f176aa380a1f31fdc5a6077dd4
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 11109, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6109b310d6e18497d03879839e4e8d16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eed7bd94a8d77db76bcaef980c0a935
    def get_inputs(self):
        return [
            paddle.uniform([22, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9567b66b6f635624eb8374601a386f08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eed7bd94a8d77db76bcaef980c0a935
    def get_inputs(self):
        return [
            paddle.uniform([145, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_05eec596b9c906f07ab9621de70ebeac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b5e63ddf040121552b163a7c1c7907d
    def get_inputs(self):
        return [
            paddle.uniform([10, 192, 1, 25], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0beaf4284940ce731297569de32f94db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eed7bd94a8d77db76bcaef980c0a935
    def get_inputs(self):
        return [
            paddle.uniform([171, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fb9a2c6f02cdf0d1098fa0c59b3dbc0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eed7bd94a8d77db76bcaef980c0a935
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d8e8d65f260729b993a1ae323af61502(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efd3a8340e42e76dd325bcdaa5b00ff7
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.1272621154785156], [2.371551752090454], [2.0648553371429443], [2.1482954025268555], [2.2486588954925537], [1.8931057453155518], [2.141450881958008], [1.8901559114456177], [1.9809638261795044], [1.9321402311325073], [2.297773838043213], [1.9171698093414307], [1.9546658992767334], [2.245638608932495], [2.2968249320983887], [2.1008729934692383], [2.1007673740386963], [2.104473114013672], [2.306422472000122], [2.227832078933716]], dtype='float32').reshape([20, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2170c30a836b00b67d365e97b7004247(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efd3a8340e42e76dd325bcdaa5b00ff7
    def get_inputs(self):
        return [
            paddle.to_tensor([[1.847316026687622], [1.9820117950439453], [2.0498132705688477], [2.144075632095337], [1.8804770708084106], [2.2597832679748535], [2.1278913021087646], [2.067511558532715], [2.102844715118408], [1.8924448490142822], [2.163910388946533], [2.282622814178467], [2.1203551292419434], [2.3010966777801514], [2.0028440952301025], [2.018085241317749], [2.1399848461151123], [1.9587953090667725], [2.0811142921447754], [1.9754807949066162]], dtype='float32').reshape([20, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9d7e3ee7300407c6a23ba7debc1796de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9668bf902e82eb0424ae466859a24798
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
class TestPrimitiveOp_fd70f563290a44b39138d2c10a2a1f2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eec651cb29c8b26e41485f015fcac29a
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 8732, 1], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_81999bda17646982763bd8c682040c89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9668bf902e82eb0424ae466859a24798
    def get_inputs(self):
        return [
            paddle.uniform([2058, 4, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7b5b3a6593cf432ee9205a61bebf8ed3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eed7bd94a8d77db76bcaef980c0a935
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_534dd0154b31494e0d661924603412a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec2127f176aa380a1f31fdc5a6077dd4
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 3024, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cdc35d198727d310963a1637f4572552(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eed7bd94a8d77db76bcaef980c0a935
    def get_inputs(self):
        return [
            paddle.uniform([11, 1280, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8e9add66b8822924a46bf19a276921cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9668bf902e82eb0424ae466859a24798
    def get_inputs(self):
        return [
            paddle.uniform([4175, 4, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2124a7f1a441a40be012200c978415e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eed7bd94a8d77db76bcaef980c0a935
    def get_inputs(self):
        return [
            paddle.uniform([1, 624, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ae139ad17346928190231cf1c52eb2b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7390a74ffd923c3cd1fbe59a6c5eed8
    def get_inputs(self):
        return [
            paddle.uniform([10, 1000, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d05d4b3be5c0a089fd4f9ba1a8e7ef48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9668bf902e82eb0424ae466859a24798
    def get_inputs(self):
        return [
            paddle.uniform([10, 1000, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5f3e448fbdfd3d96408e926dd1232127(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2, 3]
        return (lambda x, f: f(x))(paddle._C_ops.squeeze(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fc2d1c7d70c51766152bf6efaedd1899(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f3e448fbdfd3d96408e926dd1232127
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1095f6577b57fe7f655d6381e65cb03d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f3e448fbdfd3d96408e926dd1232127
    def get_inputs(self):
        return [
            paddle.uniform([1, 92, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_660d962cc11834776184cedbaa2e1053(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f3e448fbdfd3d96408e926dd1232127
    def get_inputs(self):
        return [
            paddle.uniform([22, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d596b472ea951b3946543bd0b77bdb95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f3e448fbdfd3d96408e926dd1232127
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_1a0dbadf36cf1e60437b7f3b5f6b440f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f3e448fbdfd3d96408e926dd1232127
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_ef801af349d634427a22a644d7016c1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f3e448fbdfd3d96408e926dd1232127
    def get_inputs(self):
        return [
            paddle.uniform([10, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_26a6b6f7877fa778d3d35fc4054f174a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        return (lambda x, f: f(x))(paddle._C_ops.squeeze(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, None, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a27fb89440c87d0ed8dd659bf8457108(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26a6b6f7877fa778d3d35fc4054f174a
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 3549, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_079e836b902ceca5174d2f9d6b988b4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f3e448fbdfd3d96408e926dd1232127
    def get_inputs(self):
        return [
            paddle.uniform([10, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_dccb66fd3986271254f167b5617f6e59(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        return (lambda x, f: f(x))(paddle._C_ops.squeeze(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d15e5b60aa7e5dfeff361e4916fab82e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dccb66fd3986271254f167b5617f6e59
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[3800, 1], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5124ed3103db0d5f844985d41048f3c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dccb66fd3986271254f167b5617f6e59
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[150, 1], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d94fe3f99c12f0bf39aaf40c5981a5bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f3e448fbdfd3d96408e926dd1232127
    def get_inputs(self):
        return [
            paddle.uniform([145, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cb7d64b2385cd79f78a3dbeb4940736a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dccb66fd3986271254f167b5617f6e59
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[40, 1], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1c52316312194c43a86231abeb902731(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        return (lambda x, f: f(x))(paddle._C_ops.squeeze(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_10815f38f4194cac12c7104d09ab0d7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c52316312194c43a86231abeb902731
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.0899338722229004], [2.0193772315979004], [2.2738912105560303], [1.987199068069458], [2.100257396697998], [2.1881446838378906], [1.9883549213409424], [1.8823989629745483], [2.172023057937622], [2.0769574642181396], [2.3263778686523438], [1.9662823677062988], [2.183725357055664], [2.3208024501800537], [2.061213493347168], [2.132282018661499]], dtype='float32').reshape([16, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6faa2116329de9326fc1f55620cb3471(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c52316312194c43a86231abeb902731
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.163323163986206], [1.8874764442443848], [1.9917770624160767], [1.9982393980026245], [2.045048475265503], [2.1794745922088623], [1.9257103204727173], [2.0796351432800293], [2.0372986793518066], [2.3143351078033447], [2.018178939819336], [2.2048075199127197], [2.2676215171813965], [2.1206114292144775], [1.9648325443267822], [2.009826421737671]], dtype='float32').reshape([16, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d0d1643eac43701901a1f83924f491b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f3e448fbdfd3d96408e926dd1232127
    def get_inputs(self):
        return [
            paddle.uniform([145, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c83cdfdc0c59884a52ff5597ecca1ce7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26a6b6f7877fa778d3d35fc4054f174a
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 7581, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_3028268f312e8702603d568aea845ff6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [1]
        return (lambda x, f: f(x))(paddle._C_ops.squeeze(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_06d7b26e87b3125418e085b3380c140d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3028268f312e8702603d568aea845ff6
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 18, 64, 128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_625eb3c2cf262fc050c1d647cd54014b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        return (lambda x, f: f(x))(paddle._C_ops.squeeze(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, None, 1, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c5257f98c05c105d17d1a21467f2940e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_625eb3c2cf262fc050c1d647cd54014b
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 66, 130], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8635732abdc5c4068edd58bb357b25ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26a6b6f7877fa778d3d35fc4054f174a
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4725, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d43cc6284c480462bc4eb07e6f755afb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f3e448fbdfd3d96408e926dd1232127
    def get_inputs(self):
        return [
            paddle.uniform([22, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ac677df45970d0f7fce98b6625978af9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f3e448fbdfd3d96408e926dd1232127
    def get_inputs(self):
        return [
            paddle.uniform([1, 872, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9f01757e343721c8f1d1a58644ad0a9c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        return (lambda x, f: f(x))(paddle._C_ops.squeeze(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c14d33adf8e0154b1d5dfb441f3e3c8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f01757e343721c8f1d1a58644ad0a9c
    def get_inputs(self):
        return [
            paddle.uniform([1790, 4, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_71d653e50b7f1665a52dd131d9b3953f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26a6b6f7877fa778d3d35fc4054f174a
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8400, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3d457df1a73e3912e89925c7615ff4ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f3e448fbdfd3d96408e926dd1232127
    def get_inputs(self):
        return [
            paddle.uniform([171, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_14e9e25ce6f3f34259ec91ebac600b07(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        return (lambda x, f: f(x))(paddle._C_ops.squeeze(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 768, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c9e4aec6d6be8470283b6b7b2a477283(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_14e9e25ce6f3f34259ec91ebac600b07
    def get_inputs(self):
        return [
            paddle.uniform([43, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8da0a06c1a7badb41e2d0b74d6d59505(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f3e448fbdfd3d96408e926dd1232127
    def get_inputs(self):
        return [
            paddle.uniform([10, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8f5a20d98cad1cd876524b7a569d2195(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f01757e343721c8f1d1a58644ad0a9c
    def get_inputs(self):
        return [
            paddle.uniform([5590, 4, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a7a1f62feec87df03e4a27ea017055fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c52316312194c43a86231abeb902731
    def get_inputs(self):
        return [
            paddle.uniform([36, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6bcb19e098f32be4d74c851d226f115e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2, 3]
        return (lambda x, f: f(x))(paddle._C_ops.squeeze(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1000, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_01f81e7cd40cbdaa4eac6d9cc62abf17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bcb19e098f32be4d74c851d226f115e
    def get_inputs(self):
        return [
            paddle.uniform([43, 1000, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4feb36384ea62c7806883bcc3eb55e2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dccb66fd3986271254f167b5617f6e59
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[15200, 1], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_386190eb5a772b70c558c4c1b0ac9f58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f3e448fbdfd3d96408e926dd1232127
    def get_inputs(self):
        return [
            paddle.uniform([10, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d3e188577afb1c0f81cdedde140b713f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f3e448fbdfd3d96408e926dd1232127
    def get_inputs(self):
        return [
            paddle.uniform([43, 1280, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4a5371282aeaa32b1b544fa1b814147c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bcb19e098f32be4d74c851d226f115e
    def get_inputs(self):
        return [
            paddle.uniform([10, 1000, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fd83da1fc4616a7e58ffeb52897589bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f3e448fbdfd3d96408e926dd1232127
    def get_inputs(self):
        return [
            paddle.uniform([10, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f596e668d94459e708bef040e427c9fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f3e448fbdfd3d96408e926dd1232127
    def get_inputs(self):
        return [
            paddle.uniform([22, 336, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fde33f91f805e74c2e22292587287cc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26a6b6f7877fa778d3d35fc4054f174a
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 4116, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f7a4bace1867c45b435e32d0b65d7657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f3e448fbdfd3d96408e926dd1232127
    def get_inputs(self):
        return [
            paddle.uniform([171, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e23eafbc0131aa545ec2bdaddbac3685(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f3e448fbdfd3d96408e926dd1232127
    def get_inputs(self):
        return [
            paddle.uniform([22, 1536, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ab3a5cc95480d6bde8dc0dd572d3f48e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c52316312194c43a86231abeb902731
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.006956100463867], [1.9117376804351807], [2.017982006072998], [2.205371856689453], [2.211639404296875], [2.1194374561309814], [1.9514344930648804], [1.9654366970062256], [2.209026336669922], [2.0346641540527344], [2.2032999992370605], [1.9927783012390137], [1.9959903955459595], [2.281437873840332], [2.1363065242767334], [1.9054741859436035], [1.847438097000122], [2.261545419692993], [2.121845006942749], [2.2606067657470703], [2.0738754272460938], [2.1228108406066895], [1.9678256511688232], [2.1302120685577393]], dtype='float32').reshape([24, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4f907cfbf73fb5d3d28ef5ec4d04170c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c52316312194c43a86231abeb902731
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.0476019382476807], [2.232367992401123], [2.1287949085235596], [1.9709757566452026], [2.2109475135803223], [2.2074756622314453], [1.9347628355026245], [1.9760403633117676], [2.2215323448181152], [1.88779878616333], [1.9031810760498047], [2.0182852745056152], [2.1343891620635986], [1.8733372688293457], [2.20819091796875], [2.1640238761901855], [2.239391326904297], [1.915928602218628], [2.213580846786499], [1.9242842197418213], [2.271833896636963], [1.8983896970748901], [2.3853492736816406], [1.7908287048339844]], dtype='float32').reshape([24, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_63c81e50a5341eb4bb6f1c1d99f0bb23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f3e448fbdfd3d96408e926dd1232127
    def get_inputs(self):
        return [
            paddle.uniform([171, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5f0cc50598d32aa2052bdbaa0f937d2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26a6b6f7877fa778d3d35fc4054f174a
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 6069, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a1c31aa16eea1403c6a4c189c15c04cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f01757e343721c8f1d1a58644ad0a9c
    def get_inputs(self):
        return [
            paddle.uniform([1532, 4, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_08b8bb4bd645f613ee1f2c25e336f456(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f3e448fbdfd3d96408e926dd1232127
    def get_inputs(self):
        return [
            paddle.uniform([22, 240, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6c53f935e5fc5ecf9d22af731e0dbaad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f3e448fbdfd3d96408e926dd1232127
    def get_inputs(self):
        return [
            paddle.uniform([10, 1536, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1411c516708f1b5638eb46d66306705a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c52316312194c43a86231abeb902731
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.241544723510742], [2.0525031089782715], [1.862670660018921], [2.278165578842163]], dtype='float32').reshape([4, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4eab820c6eac02f7d045616340d4fb3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c52316312194c43a86231abeb902731
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.1163506507873535], [2.1665420532226562], [2.255371570587158], [1.956946611404419]], dtype='float32').reshape([4, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7469a27d3ecb4edf49bccf4a860a187e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_625eb3c2cf262fc050c1d647cd54014b
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 70, 134], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4bd551cf2b27fdb3e7436c150d73db2b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [2]
        return (lambda x, f: f(x))(paddle._C_ops.squeeze(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 512, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a04e75ebbfd28dfbb9e96fc4eb1eed0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4bd551cf2b27fdb3e7436c150d73db2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 104, 101], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_41eabd2c33d2f174cef05383238afb29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dccb66fd3986271254f167b5617f6e59
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[2204, 1], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_890db21dea7c8c928f1192d6ca7d2a53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f3e448fbdfd3d96408e926dd1232127
    def get_inputs(self):
        return [
            paddle.uniform([22, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6e74a8d71d801784d8335edc48f5d7c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_625eb3c2cf262fc050c1d647cd54014b
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 68, 132], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_86999e67192597e3257ee29bebfe53d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bcb19e098f32be4d74c851d226f115e
    def get_inputs(self):
        return [
            paddle.uniform([11, 1000, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6f3a4867e6b422daf486bd34279cb7e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f3e448fbdfd3d96408e926dd1232127
    def get_inputs(self):
        return [
            paddle.uniform([145, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a684e2d20f08aec81265e0f0578b59a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3028268f312e8702603d568aea845ff6
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 36, 32, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_efe6ad1709512d100558a0e96792d8e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dccb66fd3986271254f167b5617f6e59
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[70, 1], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c99b21b3ba568807c4a87670c13ddcb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f3e448fbdfd3d96408e926dd1232127
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_274eec75804cca30b70b7f862d3bcaeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dccb66fd3986271254f167b5617f6e59
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[551, 1], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8bb6946fea7ca9a7d4efb061d48bc1e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dccb66fd3986271254f167b5617f6e59
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[247, 1], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8fbdfba2b511b5996e7303cca94b890e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f3e448fbdfd3d96408e926dd1232127
    def get_inputs(self):
        return [
            paddle.uniform([10, 2048, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_113bf6c60fcb7b14a3a7e007ec0b9668(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dccb66fd3986271254f167b5617f6e59
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[950, 1], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_16a555b1b74c7b7771ecc196265497c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f01757e343721c8f1d1a58644ad0a9c
    def get_inputs(self):
        return [
            paddle.uniform([2029, 4, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fc347f712ff2238ad85a5acc953dbd00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dccb66fd3986271254f167b5617f6e59
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[8816, 1], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_254ad1429e41f4e7291b0e44786f61b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f01757e343721c8f1d1a58644ad0a9c
    def get_inputs(self):
        return [
            paddle.uniform([4671, 4, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_659a82c54cda43a7e0f27d397cce0be7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        return (lambda x, f: f(x))(paddle._C_ops.squeeze(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cd9a890d4fe3ef4778a493274a8d661b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_659a82c54cda43a7e0f27d397cce0be7
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

class PrimitiveOp_1f1e04a46ee1fcc0ceb47c7216f2ffbd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        return (lambda x, f: f(x))(paddle._C_ops.squeeze(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dc34299ebd23421d2da0dabbb211e63b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f1e04a46ee1fcc0ceb47c7216f2ffbd
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 2434, 1], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8aa2ec9a475d191863d989fd465b44df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f01757e343721c8f1d1a58644ad0a9c
    def get_inputs(self):
        return [
            paddle.uniform([1040, 4, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a590a45fa8afa1413f4b08505cc87f6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26a6b6f7877fa778d3d35fc4054f174a
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 9261, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7c03be504f863e6b90b6a31d61431fff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_14e9e25ce6f3f34259ec91ebac600b07
    def get_inputs(self):
        return [
            paddle.uniform([11, 768, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4b31f4b5d86c4426b34c8a34e324eb97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_625eb3c2cf262fc050c1d647cd54014b
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 64, 128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_42d79d09088f1c7c50bdb1d31283fc1f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        return (lambda x, f: f(x))(paddle._C_ops.squeeze(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1000, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_13f820e1320c07b5b342e3349b6da993(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42d79d09088f1c7c50bdb1d31283fc1f
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e2a7a6477d6fdee2b61d09ecaf46220d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = [-1]
        return (lambda x, f: f(x))(paddle._C_ops.squeeze(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1000, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_25c3722c64d6524bbd293373f4a268a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2a7a6477d6fdee2b61d09ecaf46220d
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ce431f48a3ccbe9cc2ff06e954c326c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26a6b6f7877fa778d3d35fc4054f174a
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 2100, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ef95fa55d6110edb35c70031b0b7e944(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f3e448fbdfd3d96408e926dd1232127
    def get_inputs(self):
        return [
            paddle.uniform([1, 1248, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ec42375e64e9e3ffa0c5239f5c8e2169(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f3e448fbdfd3d96408e926dd1232127
    def get_inputs(self):
        return [
            paddle.uniform([171, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_88c47ed6fa7f4f0f9d2d926b8f0c3419(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f3e448fbdfd3d96408e926dd1232127
    def get_inputs(self):
        return [
            paddle.uniform([145, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_044f357417fd4485aec811ea7399f5d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3028268f312e8702603d568aea845ff6
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 9, 128, 256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_93237384092d3827aef6c6feb8442279(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f01757e343721c8f1d1a58644ad0a9c
    def get_inputs(self):
        return [
            paddle.uniform([2361, 4, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d960943239d2db17be5c2e7270112ef1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3028268f312e8702603d568aea845ff6
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 96, 32, 64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b4b5773b04b693dc2118948d46ae20e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f01757e343721c8f1d1a58644ad0a9c
    def get_inputs(self):
        return [
            paddle.uniform([3043, 4, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3dcd4c93a78f711c385044415abb8de5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f01757e343721c8f1d1a58644ad0a9c
    def get_inputs(self):
        return [
            paddle.uniform([3752, 4, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_80f5ea26cd84c35a0fb7189787ce1e5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3028268f312e8702603d568aea845ff6
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 24, 128, 256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_607b9359bdc34266e188b0dd2c1d37d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f3e448fbdfd3d96408e926dd1232127
    def get_inputs(self):
        return [
            paddle.uniform([1, 156, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c4eda4cf698f676db9a9bf6c112d6ba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3028268f312e8702603d568aea845ff6
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 48, 64, 128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e37839e43aecfdc79762b5ff3cf06f24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26a6b6f7877fa778d3d35fc4054f174a
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 11109, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1746f0502f9326262836e91b1a5ac469(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f3e448fbdfd3d96408e926dd1232127
    def get_inputs(self):
        return [
            paddle.uniform([22, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_92d7a96840bd220e4658f9f68b2023c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f3e448fbdfd3d96408e926dd1232127
    def get_inputs(self):
        return [
            paddle.uniform([145, 480, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3cc8f383d90451129c536d10733bea06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f3e448fbdfd3d96408e926dd1232127
    def get_inputs(self):
        return [
            paddle.uniform([171, 36, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1ad307b4dec962095a061129256b2aa5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f3e448fbdfd3d96408e926dd1232127
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7bf4335dd3319dbc255edf291cbb46d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c52316312194c43a86231abeb902731
    def get_inputs(self):
        return [
            paddle.to_tensor([[2.1272621154785156], [2.371551752090454], [2.0648553371429443], [2.1482954025268555], [2.2486588954925537], [1.8931057453155518], [2.141450881958008], [1.8901559114456177], [1.9809638261795044], [1.9321402311325073], [2.297773838043213], [1.9171698093414307], [1.9546658992767334], [2.245638608932495], [2.2968249320983887], [2.1008729934692383], [2.1007673740386963], [2.104473114013672], [2.306422472000122], [2.227832078933716]], dtype='float32').reshape([20, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1b98b96ba548fc1ab79ae8efdd788403(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c52316312194c43a86231abeb902731
    def get_inputs(self):
        return [
            paddle.to_tensor([[1.847316026687622], [1.9820117950439453], [2.0498132705688477], [2.144075632095337], [1.8804770708084106], [2.2597832679748535], [2.1278913021087646], [2.067511558532715], [2.102844715118408], [1.8924448490142822], [2.163910388946533], [2.282622814178467], [2.1203551292419434], [2.3010966777801514], [2.0028440952301025], [2.018085241317749], [2.1399848461151123], [1.9587953090667725], [2.0811142921447754], [1.9754807949066162]], dtype='float32').reshape([20, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_edaedd2a069e9d6642db78cef15222ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_659a82c54cda43a7e0f27d397cce0be7
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
class TestPrimitiveOp_c5e471e06dfe3f416c1e0f1c2e2f87e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f1e04a46ee1fcc0ceb47c7216f2ffbd
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 8732, 1], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7f03abf7bc2c0c041be83daa901b984c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f01757e343721c8f1d1a58644ad0a9c
    def get_inputs(self):
        return [
            paddle.uniform([2058, 4, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a91e63a1a4a076d587fd4f06fdee9ea8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bcb19e098f32be4d74c851d226f115e
    def get_inputs(self):
        return [
            paddle.uniform([22, 1000, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_43162c866c5a3b0a1cf77ddfaddb63be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26a6b6f7877fa778d3d35fc4054f174a
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 3024, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1a765e982b081c98f720941f1748d19c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f3e448fbdfd3d96408e926dd1232127
    def get_inputs(self):
        return [
            paddle.uniform([11, 1280, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_44b6d34596d519878ded0969bf6a69e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f01757e343721c8f1d1a58644ad0a9c
    def get_inputs(self):
        return [
            paddle.uniform([4175, 4, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_77fe46bf5d8e792a88505f568b41a5e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f3e448fbdfd3d96408e926dd1232127
    def get_inputs(self):
        return [
            paddle.uniform([1, 624, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3aa1131212265eb35588fac48f1772ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42d79d09088f1c7c50bdb1d31283fc1f
    def get_inputs(self):
        return [
            paddle.uniform([10, 1000, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fe2ff33a8c4f583b512a84e34aa78bf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2a7a6477d6fdee2b61d09ecaf46220d
    def get_inputs(self):
        return [
            paddle.uniform([10, 1000, 1], dtype='float32', min=0, max=0.5),
        ]


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