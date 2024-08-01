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
class PrimitiveOp_cb60f9c200b2237364d4f861ea777605(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e4b2497f64b1d19f05319266a53d3049(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 96, 96], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b0912a935dc7f46d252013b10096c6dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.168771266937256]], [[2.9708309173583984]], [[2.80727219581604]], [[2.2072644233703613]], [[2.4070839881896973]], [[3.2571425437927246]], [[2.67838191986084]], [[3.602384567260742]]]], dtype='float32').reshape([1, 8, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_97f40fb9308b821e733e352d237cd143(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 96, 96], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_386100d4a1bc5aab091e511ead60c242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 48, 48], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_041eab712503cac4ca51484ac0332434(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[213.1594696044922]], [[190.10787963867188]], [[231.5343017578125]], [[190.83932495117188]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_28a2cdca5623b01f5daa417e8cb362b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 48, 48], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_50d58d49813ca6b7f283d7569adf46db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[38695.4296875]], [[38113.9765625]], [[40714.5625]], [[35538.03515625]], [[40491.65234375]], [[40164.72265625]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8082188590dd89960ac8badec558bc5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 24, 24], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c6e28ac095442b2392607ff15c830516(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[15250180.0]], [[16970544.0]], [[18948580.0]], [[15148478.0]], [[18938752.0]], [[20055974.0]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_075738bb0d5742f15f96bd739c473061(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 24, 24], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7918807b3e7ab7de7a276ef5a3efada9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[26926512128.0]], [[23695814656.0]], [[24884244480.0]], [[24262090752.0]], [[19979395072.0]], [[22436022272.0]], [[23215804416.0]], [[19340918784.0]], [[21946474496.0]], [[20572753920.0]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f6bfcb4a1960faacedd38b8005b84bce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 12, 12], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_10941b7dee4e14e3a338229936348750(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7089318002688.0]], [[7096913887232.0]], [[8829345464320.0]], [[6649573015552.0]], [[6078255333376.0]], [[9627587575808.0]], [[6489096847360.0]], [[7387590164480.0]], [[8540734357504.0]], [[5733305810944.0]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a577c8778766e40194d2c71fe5a6df06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 12, 12], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3ba69421b239f3a4d2f8aa03558a4a84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.854335809172275e+16]], [[1.7774536397160448e+16]], [[1.7608366259961856e+16]], [[2.0639426686222336e+16]], [[1.767479974035456e+16]], [[1.954803684160307e+16]], [[1.7916655791243264e+16]], [[1.6424000037060608e+16]], [[1.886651143107379e+16]], [[1.884789274784563e+16]], [[1.921506950198067e+16]], [[1.8931578038124544e+16]], [[1.8714218333208576e+16]], [[1.905922876112896e+16]], [[2.2127409515986944e+16]], [[1.9033753162612736e+16]], [[1.9241771313659904e+16]], [[1.8564850108071936e+16]], [[1.948297238203597e+16]], [[1.861284207263744e+16]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_31f05dd4d117ac801e41834299abe879(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.630484040823341e+19]], [[3.6839634067890635e+19]], [[3.527057380044964e+19]], [[3.2630087830252552e+19]], [[4.023019807446349e+19]], [[3.5937238486695805e+19]], [[3.880627773993543e+19]], [[3.881598862663195e+19]], [[3.9390144802563555e+19]], [[3.113494553052368e+19]], [[3.3738758389887926e+19]], [[3.6706936208557605e+19]], [[3.9293937535133155e+19]], [[3.920527731551581e+19]], [[3.283296531776327e+19]], [[3.814969337629272e+19]], [[3.932807077410583e+19]], [[3.279151592841937e+19]], [[3.5099487592144437e+19]], [[3.434870806735015e+19]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9b3b29b9d4082d656c328f1c57835b1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.23141102873074e+22]], [[5.6962928706466146e+22]], [[6.281159591654525e+22]], [[5.426881686257796e+22]], [[6.62782102045163e+22]], [[5.719542703722915e+22]], [[6.114565136398612e+22]], [[6.106090712979789e+22]], [[5.988488666270226e+22]], [[5.6058187066524805e+22]], [[6.918643218188893e+22]], [[6.7972180657556555e+22]], [[5.844374829274258e+22]], [[5.078176974309753e+22]], [[5.356730015582171e+22]], [[6.194313977880163e+22]], [[5.879900574214845e+22]], [[5.5581485549567265e+22]], [[5.497007686415545e+22]], [[5.510250521119828e+22]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_62c5770d8082589d9ae8ddddcc4f67fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.821779625276395e+26]], [[3.353424750722504e+26]], [[3.168796085442204e+26]], [[2.8879034551481327e+26]], [[3.107271397802924e+26]], [[3.129324111408162e+26]], [[3.0188801342249297e+26]], [[3.0325847737996107e+26]], [[3.3450628416338915e+26]], [[2.8102961581556293e+26]], [[3.273419377000418e+26]], [[3.0883162614625428e+26]], [[3.3082542085092114e+26]], [[3.104119956045371e+26]], [[3.169809549561614e+26]], [[3.25507298321447e+26]], [[3.399893205848823e+26]], [[2.9835197549774766e+26]], [[3.181134005748464e+26]], [[2.822869827851151e+26]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2e83d4fce6ebfae58262238e83813c6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 12, 12], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_639accb4ffdca83e78fd8b7e63068e4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.3643148126548069e+30]], [[2.4543545498896123e+30]], [[2.4150363532268308e+30]], [[2.122631954079366e+30]], [[2.2961138670041574e+30]], [[2.353872564772158e+30]], [[2.194664892346739e+30]], [[2.4385765567863668e+30]], [[2.12105067910731e+30]], [[2.3861554176276922e+30]], [[2.044368514369154e+30]], [[2.2043763445714306e+30]], [[2.3333761341949567e+30]], [[2.3015955900174724e+30]], [[2.0953262488231852e+30]], [[2.2908567019618357e+30]], [[2.269467479666849e+30]], [[2.3490779649715664e+30]], [[2.0903125312177884e+30]], [[2.1787747713737242e+30]], [[2.1739327212347127e+30]], [[2.1663988466426018e+30]], [[2.345596863073986e+30]], [[2.398099000262575e+30]], [[2.358719752845903e+30]], [[2.2572679069896628e+30]], [[2.3374963045039308e+30]], [[2.104734714014336e+30]]]], dtype='float32').reshape([1, 28, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_96efe6eba17a6041f5c1edf49c80ec97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.1115171868078946e+34]], [[2.212368200400349e+34]], [[2.1473912036183378e+34]], [[2.260404730508763e+34]], [[2.296651367271031e+34]], [[2.304113917415851e+34]], [[2.1018805663660815e+34]], [[2.0387568896168846e+34]], [[2.342834701140635e+34]], [[2.0004917916205617e+34]], [[2.456818525433852e+34]], [[2.200725126742862e+34]], [[2.2526844412477636e+34]], [[2.224148932990204e+34]], [[2.2247723595939883e+34]], [[2.0508742181034216e+34]], [[2.140541433792964e+34]], [[1.9629482888731382e+34]], [[2.0138078174471388e+34]], [[2.3299175395947156e+34]], [[2.352923417284795e+34]], [[2.3292512802655723e+34]], [[2.145086654441204e+34]], [[2.464766843250088e+34]], [[2.2469542643939195e+34]], [[2.2768368990022293e+34]], [[2.2937280956622626e+34]], [[2.208974511576652e+34]]]], dtype='float32').reshape([1, 28, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1fcc887e031419d727d5dba6511c8c2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.496062578572521e+38]], [[1.449623162247616e+38]], [[1.4040660307404698e+38]], [[1.4381327715589552e+38]], [[1.4410262601130122e+38]], [[1.5174655913067744e+38]], [[1.4237032568946293e+38]], [[1.3989888365564357e+38]], [[1.3946199041157611e+38]], [[1.4864938447817582e+38]], [[1.3905615967781665e+38]], [[1.3465895355815457e+38]], [[1.377639470795584e+38]], [[1.4136798928926007e+38]], [[1.375850967916734e+38]], [[1.3534947833431569e+38]], [[1.5114691983195028e+38]], [[1.4758988210650507e+38]], [[1.3900932759404181e+38]], [[1.4132580187728447e+38]], [[1.4416011650132277e+38]], [[1.3827135212061295e+38]], [[1.3572852614619353e+38]], [[1.412253938085416e+38]], [[1.418896630054708e+38]], [[1.3987816517423344e+38]], [[1.3919276170649724e+38]], [[1.4126444758823343e+38]]]], dtype='float32').reshape([1, 28, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_aba219735175354d35c4fff940853712(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 6, 6], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1881558ceddc82769ea5d5a5bbb0e34e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 28, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ac0f4694975c8e128ddaf1c9ffdecfc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 1152, 6, 6], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f2e7beee64e846114758588dcb9fc00a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_af8ddb0337bdb5393a51d1c3a36fa10a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 1280, 6, 6], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a313ee4fed4cf271a5112f50bb8ddf88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 112, 112], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ce71d0658d0d6296eafb480a58c13f6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 112, 112], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_75765ae7916920c84bcb0b114b4f09e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 162, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7f85100d68e7fa59230f2f759c4b5e11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 228, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_07a3c6f25fdd18a32a7462c82a8b1c4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 300, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eb0a224864c8e2f18d65b1221fdaa5db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 366, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b0100da7452e57244315bf785a73bede(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 432, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1f4d9cbe05a4d464de117f28f5c091e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 504, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0f3bc4e1909da1977a6c505a31379129(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 570, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_56327f6d550e008b1029e34d8132703a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 636, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7284dd3ee7b7e76713bad812a89be39a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 702, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_97561ed07fed7d3636ed22df28246365(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4376b1fffc215b9af535c4ae1262e580(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 840, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_beab909d0eb20820023a9b8326423f2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 906, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6ed608ebde16c040d022bb240bf0a11a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 972, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4f18e66da62f4dfb380934b26a7880fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 1044, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_187c89aa996c36a61a6a20521cfa35f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 1280, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a7f3374ece94d8aa2197285ccc6281fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_860f017d6b136e8584ff686711a056e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_d302e02db1eca90cbd40f16885e9059d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_87d95732e7be8db5b34ec0bf43236518(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 160, 160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3fd498e78dfb1430e4a140e8984fef3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 160, 160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d7ef78ad82e5d5a5a55db93c66d3f3cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_c0458ec0b888a5fca12cb193f756fb19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_c771ddf27147f9316e98a3dd4c95d968(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 80, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b964b234b0b58fdf63a930e3dc9cdb5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_0f16744fa8a74cfad4e7203413765d04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_f15fe6d4aa848f4608416bdad89575ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a1dbd64b80f5c66a72e717d29a5f56be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_4476535a2df32edd71c5a96743591dec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_29b0e5ceb0df64deb47a82410b99c10c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 20, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dd37862d66ae7629153669757b0a839a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8456f9525585f452ae5381e1d0b987be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7797bd0267c521e6e4889188b480f1f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[19006174.0]], [[19235500.0]], [[17557304.0]], [[19838796.0]], [[19611992.0]], [[16834778.0]], [[21152790.0]], [[19725676.0]], [[20947212.0]], [[21822048.0]], [[18394660.0]], [[17615450.0]], [[16969952.0]], [[22496512.0]], [[17739876.0]], [[23222528.0]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cf88881ffb0c0f4b11f9c90be63da20d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ed0018b889d1462912e3d9ef5a390d60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7190799360.0]], [[6596408832.0]], [[7201764864.0]], [[8452954624.0]], [[7108325376.0]], [[6885442560.0]], [[8143262720.0]], [[6375717376.0]], [[8458877952.0]], [[6819305472.0]], [[6888590336.0]], [[7887463424.0]], [[7887997952.0]], [[6318848512.0]], [[7731600384.0]], [[7861496832.0]], [[8238494720.0]], [[6979163136.0]], [[6549292032.0]], [[7918797824.0]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0a721d70e744f683db2b71602e3b6a38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1677949403136.0]], [[1673058320384.0]], [[1420855476224.0]], [[1648670015488.0]], [[1598518984704.0]], [[1477845188608.0]], [[1612381159424.0]], [[1935770648576.0]], [[1515699437568.0]], [[1627987247104.0]], [[1952477347840.0]], [[1728554467328.0]], [[1562357923840.0]], [[1374748934144.0]], [[1324509822976.0]], [[1291124998144.0]], [[1614997094400.0]], [[1541365563392.0]], [[1449140158464.0]], [[1271105847296.0]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_140f75e161b1129c11fc287be84156d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[369381112545280.0]], [[391802150453248.0]], [[337451453251584.0]], [[250776597299200.0]], [[303183452700672.0]], [[335065733136384.0]], [[293844549632000.0]], [[369366650585088.0]], [[396570134577152.0]], [[349206376087552.0]], [[356253645668352.0]], [[322784844578816.0]], [[397283770236928.0]], [[356633817382912.0]], [[318540477366272.0]], [[331936547471360.0]], [[407114782605312.0]], [[294153820831744.0]], [[326793827450880.0]], [[281171443318784.0]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_35ebec7af544d7e114d49386d6cc88ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6b77c363180c93d78b287d5c1646cb4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.5312278089826304e+17]], [[4.292216455346258e+17]], [[4.313428096230359e+17]], [[2.614324087446569e+17]], [[4.239283904000819e+17]], [[2.525378578921554e+17]], [[3.968344997463327e+17]], [[3.8454491468529664e+17]], [[3.526352661811036e+17]], [[2.9179368717890355e+17]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a35a2665489fe61029f1bbef0851b509(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4dad9f673af1002e429d5eaf8c308de4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.8778166785037514e+21]], [[1.665290404713254e+21]], [[1.4771042573213457e+21]], [[1.4230918833028496e+21]], [[1.674811295800492e+21]], [[1.9973043542297967e+21]], [[1.8161486073311904e+21]], [[1.8495689757409768e+21]], [[1.274819591295954e+21]], [[2.1021084655832657e+21]], [[1.9342070938629875e+21]], [[1.2905325096583614e+21]], [[1.8269430319505557e+21]], [[1.3110839828704248e+21]], [[1.5834585921090486e+21]], [[2.1872249504281961e+21]], [[1.4959183262396627e+21]], [[1.9333789944815047e+21]], [[1.955100840646731e+21]], [[1.7115378692867217e+21]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_392074e033ce4dea21351de8a24cd870(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.390321905160308e+24]], [[3.0724640404135854e+24]], [[3.2631428457567503e+24]], [[3.384238803071626e+24]], [[3.7169771374479327e+24]], [[2.550668196559967e+24]], [[3.1074592433036693e+24]], [[3.13905131014253e+24]], [[3.1042699741915507e+24]], [[2.766700034019813e+24]], [[3.500168519124358e+24]], [[3.197247905391322e+24]], [[2.7528551761317414e+24]], [[3.3659693206792496e+24]], [[2.6314568651522827e+24]], [[3.3629065847022615e+24]], [[3.0457563255289917e+24]], [[3.685714229698637e+24]], [[3.2631978977585953e+24]], [[3.40785092371635e+24]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1292a5edad71a42fa71add44eb138800(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.191811218787006e+27]], [[7.095622454217924e+27]], [[7.310751039936671e+27]], [[6.747469527043222e+27]], [[5.513372903779087e+27]], [[6.093075855820906e+27]], [[7.568960594487018e+27]], [[8.52792093083877e+27]], [[8.461526228977054e+27]], [[6.218124710583104e+27]], [[7.604449178605783e+27]], [[5.562950668299494e+27]], [[6.126870290963941e+27]], [[5.684747583442426e+27]], [[5.406959097454102e+27]], [[5.535262843314619e+27]], [[5.514021638874671e+27]], [[8.292876354774331e+27]], [[9.46577937256182e+27]], [[9.092308067785022e+27]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ea32e1a74d453c0eac9b88baafa14b97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fd62e34de40628c4a98f4de802faae99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 360, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9b9fde159d356366a7118507e87e2da7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_102bab94e6c5131693993cf258ba1cd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6af0effdfa975472eab4d129bb5d7c96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_336b8da4c393d4ad70de5105e4fd4c62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 1200, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3892eff1da8f341a325b526594f8d631(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d466ead746e3ac0bca6e59352446b298(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_13e7355d01b273bc100e4a98fb400739(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 56, 56], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_26345792f25d3358c8278363fd2505fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 28, 28], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0ec7834994122efcdb1cf52ead3c89d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_87d1450e307fb6326f8cf16e542c8b76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 28, 28], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ed56d4b998c9b7f7e336fb63133841c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9fc8f81c8a5d24ea255f936ebb79a489(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0329c1af0b33b5634954bc90d1e4c662(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 10, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_180c694c1cff47afefa8f8fcf9ce11f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0fd9abadf5c41225203ca33bb9ba251d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_9bca7df20372a0bf42e81677b3b2202e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 360, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a8e45f6b10e91abe33bd1e77bc911745(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5e3213994895aae475a0251b62868228(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a6839c744977e1f68d39cf67e532c612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ba3a14e12f4e3cc69f40990908c98e13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 1200, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5c85162b49e61d0c783efce084a789f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3ddf95edf39225de98ae5a876f7e3573(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 96, 96], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_66b90a3d20b09a35ecc6ce52248bedeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.203125]], [[2.28515625]], [[2.30078125]], [[3.12890625]], [[2.748046875]], [[2.375]], [[2.8984375]], [[3.3671875]]]], dtype='float16').reshape([1, 8, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d37865472d01391a0a3a19a6f1112a38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 96, 96], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a24b8d07330381128716201acaf71728(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 48, 48], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c20d35c14cfb709ea415067c713452f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[79.875]], [[79.9375]], [[77.75]], [[65.75]]]], dtype='float16').reshape([1, 4, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6a2f0bf13735541651d5d169451e8ad8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 48, 48], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_34c6d773556adba47f69295d0852e9c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[12784.0]], [[14912.0]], [[12584.0]], [[12112.0]], [[14512.0]], [[12680.0]]]], dtype='float16').reshape([1, 6, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_416c24a7d4e294d513fa88c0d034b97e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 24, 24], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e48f2900a50260824924a6c4d55f6987(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0ea7438f7e93d13c73aa8f638b2b0abc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 24, 24], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_73ee42af3a26591981763da69204fef6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 12, 12], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5d96dbf7e2bd105af259bbb110cb426c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 12, 12], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a84935f3400b3bfc930e99007c77574b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 12, 12], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e76edc0387c1d1013aea9829934a563f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 28, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ec3a0ee43bbfe5a5c6448bc9f3f09d5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 6, 6], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e9479fb731a02e5003d885f416b958ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 1152, 6, 6], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e365650093283f1589b499bb9a665c2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_04c83d3bd47dcd3591d968f25b5187e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 1280, 6, 6], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b6a841ef81dce952f91499dfdc10aca9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 112, 112], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_14c4e71078e6507a54e9685f98057f4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 112, 112], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cf0cee8921bf34a59fb0b90958587320(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 162, 56, 56], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_76ab5156587107c28908969da80e84b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 228, 56, 56], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_03709c2f9acc93fc68c204fa3a663e02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 300, 28, 28], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9b518df91e7c09e5ca00f9fb21cc8039(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 366, 28, 28], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c109e26d46030ad36ddff55763a4c869(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 432, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f5e045d51bd38888c824fce34b805520(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 504, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e265e537ff02ea129c13eecfa9279097(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 570, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6700b898a1f72bc3d9af422c40c61356(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 636, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_197dec4f49a5100a42191edf2ff7f043(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 702, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_05f11788c018776b11db7fad519360aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_06f48e9f4bb92e959dee3202e0b31d95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 840, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_42915f81b4f0091b31dbdab3a6de2816(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 906, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9bef1b4450cb30012a6f398535e64245(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 972, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5d02c869b8c62cbd5728a19ea5287b40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 1044, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b2f49e34dd1f2b0e7eb383499efa6a79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 1280, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a77f5239830feea515b39b15c6a02286(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.922877788543701]], [[3.5791587829589844]], [[4.144195556640625]], [[4.144674301147461]], [[3.3976290225982666]], [[3.6537511348724365]], [[3.263800859451294]], [[3.7594494819641113]]]], dtype='float32').reshape([1, 8, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_219417035c760190f5a04040e8bdc37f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3541d5daf233e85a3f9984220fb7f4ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[380.4045715332031]], [[292.51519775390625]], [[319.36065673828125]], [[269.7727355957031]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8d0b7f03b92dbf1090564ec20999be9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2999f14cfdb46041ff5f8444f8ddc19b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[29749.802734375]], [[34152.0859375]], [[29265.646484375]], [[27124.87890625]], [[30676.791015625]], [[25434.556640625]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9094273acae911448a3e559f65fdc1a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_258f6b8cbc8bc269cf992d90c0c39648(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[22595958.0]], [[23814368.0]], [[22048644.0]], [[24586742.0]], [[19240024.0]], [[20879972.0]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_41af3462f4c956a9335105091ebaa1f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[17820053504.0]], [[17573085184.0]], [[12539602944.0]], [[16684223488.0]], [[15716587520.0]], [[13704207360.0]], [[16981073920.0]], [[16666883072.0]], [[13481972736.0]], [[13344173056.0]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a30b1c5b90fa04e3296ea170863958a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[11998731436032.0]], [[12011637309440.0]], [[11930318143488.0]], [[12494372339712.0]], [[11370160455680.0]], [[13144015503360.0]], [[11300576952320.0]], [[10547793756160.0]], [[12579927752704.0]], [[12332767903744.0]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_00f7d370f353da8746031658c2ab3bc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.3558145527840768e+16]], [[1.2930531620552704e+16]], [[1.2398914527297536e+16]], [[1.4013161879371776e+16]], [[1.2977771965841408e+16]], [[1.4001563320188928e+16]], [[1.4354497962770432e+16]], [[1.3874335651463168e+16]], [[1.2451473115840512e+16]], [[1.3568060459843584e+16]], [[1.4111631520825344e+16]], [[1.361962583719936e+16]], [[1.3106773456060416e+16]], [[1.3741105765941248e+16]], [[1.3777262948122624e+16]], [[1.3786289895636992e+16]], [[1.2567663792357376e+16]], [[1.407670699425792e+16]], [[1.3706006219456512e+16]], [[1.372359088930816e+16]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d39e91c24e8f775cd2fd0097e15478cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.4546745543662502e+19]], [[1.515836977733383e+19]], [[1.4809108809259155e+19]], [[1.5719047040220529e+19]], [[1.6000018440605663e+19]], [[1.428553346674013e+19]], [[1.587737341608539e+19]], [[1.5447489658392412e+19]], [[1.5669181988877631e+19]], [[1.5068736489925902e+19]], [[1.352543118285393e+19]], [[1.4825126494652596e+19]], [[1.436059052849863e+19]], [[1.5100369439457018e+19]], [[1.4507147731899777e+19]], [[1.5027874239791235e+19]], [[1.4982748083564052e+19]], [[1.5746264351054496e+19]], [[1.5435203715463643e+19]], [[1.6264833617132257e+19]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_83269bf9eb8293cb19ce34413531143b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.286318036848084e+22]], [[5.697436784951967e+22]], [[5.176352293666691e+22]], [[5.826673430938879e+22]], [[5.557613977680958e+22]], [[5.964559690370118e+22]], [[5.316860548800986e+22]], [[5.359184027019125e+22]], [[5.915843352480926e+22]], [[5.509547959577958e+22]], [[5.105060761925379e+22]], [[5.33430929519727e+22]], [[5.102122163168519e+22]], [[6.224474584584664e+22]], [[5.946090878658235e+22]], [[5.608352431802839e+22]], [[6.118477863754872e+22]], [[4.8608278519733e+22]], [[5.67184778222921e+22]], [[5.571804820106802e+22]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_54cc471dd99a6c285db42bedd85713e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_46606e6700f7eb97a8649ea2652a7b98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.6257573538966114e+26]], [[3.5318741256729925e+26]], [[3.5369484560327885e+26]], [[3.3819795726788434e+26]], [[3.5437685862517204e+26]], [[3.625693897096998e+26]], [[3.411988735937954e+26]], [[3.440398566485874e+26]], [[3.682524626239282e+26]], [[3.391286692933793e+26]], [[3.612572359102487e+26]], [[3.3940304616473164e+26]], [[3.470999870229751e+26]], [[3.3713424421761796e+26]], [[3.4559281424517673e+26]], [[3.416584188821597e+26]], [[3.591479614058844e+26]], [[3.704010286931695e+26]], [[3.415231304611231e+26]], [[3.477954661680421e+26]], [[3.8338384717224625e+26]], [[3.627645931554878e+26]], [[3.3974958669890035e+26]], [[3.3403939707088356e+26]], [[3.466741623827776e+26]], [[3.727230310501917e+26]], [[3.4337994282609454e+26]], [[3.5828771594275106e+26]]]], dtype='float32').reshape([1, 28, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_84662957223d6ff0b1baf7895d4695b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.3777476407837274e+30]], [[2.2989069389946495e+30]], [[2.2962497200431365e+30]], [[2.4124232600677338e+30]], [[2.429969458297893e+30]], [[2.2865320720736193e+30]], [[2.3708974137426085e+30]], [[2.3604315918064772e+30]], [[2.1471603033807095e+30]], [[2.4524716479255625e+30]], [[2.4047444654929965e+30]], [[2.261522570296069e+30]], [[2.1575146019099813e+30]], [[2.3544863968570674e+30]], [[2.3048137505492866e+30]], [[2.4102824035569237e+30]], [[2.3344611451180608e+30]], [[2.3898656109680895e+30]], [[2.240436939497533e+30]], [[2.4122933005421252e+30]], [[2.216803044186977e+30]], [[2.3989917919803603e+30]], [[2.3858102693061922e+30]], [[2.2991520487045764e+30]], [[2.400677034572903e+30]], [[2.2883199222451018e+30]], [[2.288468166773732e+30]], [[2.3874943029729154e+30]]]], dtype='float32').reshape([1, 28, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f28cbdd5289d2bdcd721bc80dc86bfaf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ef40f29e19d3c6c7f790a4d41fb4904b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.056422045350911e+34]], [[5.55795067417066e+34]], [[5.133265440173496e+34]], [[5.370417613499397e+34]], [[5.555486183140451e+34]], [[4.787142356949461e+34]], [[5.364995931303342e+34]], [[5.267108050988922e+34]], [[5.3585413119385085e+34]], [[5.582729281996997e+34]], [[4.816064102499278e+34]], [[5.286069330982648e+34]], [[5.021515602475174e+34]], [[5.091482983143552e+34]], [[5.080955541049469e+34]], [[5.248341870345387e+34]], [[5.218366885410146e+34]], [[5.089253700720807e+34]], [[5.390759444224934e+34]], [[4.960452476921367e+34]], [[4.998571126611043e+34]], [[5.309133639442582e+34]], [[4.96736018234058e+34]], [[5.591924205432793e+34]], [[5.434502303101091e+34]], [[5.347184450018104e+34]], [[5.750695957583331e+34]], [[5.187593676737575e+34]]]], dtype='float32').reshape([1, 28, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7573d57f6043cb7ed3818fb2532d72ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 1152, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1ebc7055cad9492c52c796ec07a0f796(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.99609375]], [[6.70703125]], [[6.203125]], [[5.5859375]], [[6.6796875]], [[6.1328125]], [[4.98046875]], [[5.2734375]]]], dtype='float16').reshape([1, 8, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4cd744e78bb1d9c9b3f7f3d4aee5e9f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 56, 56], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9745cdfd90a5e802450ab177497e05db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1193.0]], [[1526.0]], [[1433.0]], [[1565.0]]]], dtype='float16').reshape([1, 4, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1a3ffb4c72827fcf7a0a94c1f17c93cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 56, 56], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0cbfeba32f7407aee502d1ca69c8c560(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 28, 28], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_164b84a3177d009f46793f85d26b8f46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_32aa854ae4b34196f5d5d4b9e1c2722b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_caf72d9b1a3dbbe94f1dbc2829a7cb87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 1152, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_29fb4c5dc6dc3c263a2cf62f917669b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_7bbba99488ffc1abc0c8015c8e4519c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_ce84fe13913165f1951e29ffdc2636fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_01fd8b59b3ff87468c134b1ffa76dc17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 160, 160], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e1cc232f60da72bfaf8ecabef8fef83b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 160, 160], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cd5c2569656e783d0b7291238ce1a230(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_2aeef4a9ca87c23856f250680b4ed183(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_7bae603c02e442a85de29342119501e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 80, 80], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3c982643560eddcf9eb87e453d78bfee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_ec01b0c56a60ccf9f59bebe7c4b305e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_dd2966ce08f91cdd7f73e0d4b4ea4227(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_80d66bccef3f141a0564ea3060732436(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_872196f136918023d71f0dfc40116de7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_3f6dce2a64c9f00c6e6e2bdf1154ef31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 20, 20], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_30cee020f0c23037ff589bb4984c284d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 96, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e2684c983afd6a4b818be4bee21daaf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30cee020f0c23037ff589bb4984c284d
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 96, 96], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_29007ee7b2e66f3b636fcfe61bbce4a2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8f4f1d9bce18291c94891140ef769f01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29007ee7b2e66f3b636fcfe61bbce4a2
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.168771266937256]], [[2.9708309173583984]], [[2.80727219581604]], [[2.2072644233703613]], [[2.4070839881896973]], [[3.2571425437927246]], [[2.67838191986084]], [[3.602384567260742]]]], dtype='float32').reshape([1, 8, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e3d9939d54ec203c7b968960029317e8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 96, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_569a3a7a5478c8ccf559c03f80b5b6b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3d9939d54ec203c7b968960029317e8
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 96, 96], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ac77e68e6c5ab49d396aa9c9e2f8174b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 48, 48], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_38e9b4e2c477a1db7bab666948b9f349(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac77e68e6c5ab49d396aa9c9e2f8174b
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 48, 48], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_10b285d1685c7e1ec0629a08f4f1c2b5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c30f0f0a9fea375e0e6fb35537172844(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10b285d1685c7e1ec0629a08f4f1c2b5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[213.1594696044922]], [[190.10787963867188]], [[231.5343017578125]], [[190.83932495117188]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d1c69918082585ed516f637b12ba849a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, 48, 48], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3c77c5770ebec4b6d59f2959c1851b19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1c69918082585ed516f637b12ba849a
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 48, 48], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f8d07bd6ef969a87be053df3f69b43d8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 6, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c3d66dd2a68a9b89fc4aea3169c84feb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8d07bd6ef969a87be053df3f69b43d8
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[38695.4296875]], [[38113.9765625]], [[40714.5625]], [[35538.03515625]], [[40491.65234375]], [[40164.72265625]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2a658ede0cce07de6004443e6b0407f7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, 24, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c3dd45dd374a3b0b44a08a068338ba93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a658ede0cce07de6004443e6b0407f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 24, 24], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_34049841b42413fa190b213c68c3342f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8d07bd6ef969a87be053df3f69b43d8
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[15250180.0]], [[16970544.0]], [[18948580.0]], [[15148478.0]], [[18938752.0]], [[20055974.0]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b56e6ce950d861d165eed15e3874aac4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, 24, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_754b816cb64078a5c0ea52a1ff5119b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b56e6ce950d861d165eed15e3874aac4
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 24, 24], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ff33d5afbedb70bb059ced37dbdb332d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 10, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9922c62c05fde04ac2996e4ce29f6a53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff33d5afbedb70bb059ced37dbdb332d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[26926512128.0]], [[23695814656.0]], [[24884244480.0]], [[24262090752.0]], [[19979395072.0]], [[22436022272.0]], [[23215804416.0]], [[19340918784.0]], [[21946474496.0]], [[20572753920.0]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f334fd59e8671ada5bd37c30d1b7aa3e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, 12, 12], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_def7db6db61a6350ad7333905c31bb48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f334fd59e8671ada5bd37c30d1b7aa3e
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 12, 12], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0179bd378cac28955de75e34f65886c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff33d5afbedb70bb059ced37dbdb332d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7089318002688.0]], [[7096913887232.0]], [[8829345464320.0]], [[6649573015552.0]], [[6078255333376.0]], [[9627587575808.0]], [[6489096847360.0]], [[7387590164480.0]], [[8540734357504.0]], [[5733305810944.0]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_40f5dd261d5360f4326d884ef14215a0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 480, 12, 12], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a30aa871ea2708bc05d3fd98238b117e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40f5dd261d5360f4326d884ef14215a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 12, 12], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4f85ff14ecb1201f01ba8ba107f90885(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 20, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_93164f0fc0603b31b24f12733e813ee3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f85ff14ecb1201f01ba8ba107f90885
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.854335809172275e+16]], [[1.7774536397160448e+16]], [[1.7608366259961856e+16]], [[2.0639426686222336e+16]], [[1.767479974035456e+16]], [[1.954803684160307e+16]], [[1.7916655791243264e+16]], [[1.6424000037060608e+16]], [[1.886651143107379e+16]], [[1.884789274784563e+16]], [[1.921506950198067e+16]], [[1.8931578038124544e+16]], [[1.8714218333208576e+16]], [[1.905922876112896e+16]], [[2.2127409515986944e+16]], [[1.9033753162612736e+16]], [[1.9241771313659904e+16]], [[1.8564850108071936e+16]], [[1.948297238203597e+16]], [[1.861284207263744e+16]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1aacf7753af2b3c00a1a601ea7b52fc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f85ff14ecb1201f01ba8ba107f90885
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.630484040823341e+19]], [[3.6839634067890635e+19]], [[3.527057380044964e+19]], [[3.2630087830252552e+19]], [[4.023019807446349e+19]], [[3.5937238486695805e+19]], [[3.880627773993543e+19]], [[3.881598862663195e+19]], [[3.9390144802563555e+19]], [[3.113494553052368e+19]], [[3.3738758389887926e+19]], [[3.6706936208557605e+19]], [[3.9293937535133155e+19]], [[3.920527731551581e+19]], [[3.283296531776327e+19]], [[3.814969337629272e+19]], [[3.932807077410583e+19]], [[3.279151592841937e+19]], [[3.5099487592144437e+19]], [[3.434870806735015e+19]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_710260bfd4d1267803558999b3334457(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f85ff14ecb1201f01ba8ba107f90885
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6.23141102873074e+22]], [[5.6962928706466146e+22]], [[6.281159591654525e+22]], [[5.426881686257796e+22]], [[6.62782102045163e+22]], [[5.719542703722915e+22]], [[6.114565136398612e+22]], [[6.106090712979789e+22]], [[5.988488666270226e+22]], [[5.6058187066524805e+22]], [[6.918643218188893e+22]], [[6.7972180657556555e+22]], [[5.844374829274258e+22]], [[5.078176974309753e+22]], [[5.356730015582171e+22]], [[6.194313977880163e+22]], [[5.879900574214845e+22]], [[5.5581485549567265e+22]], [[5.497007686415545e+22]], [[5.510250521119828e+22]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_56eec898886d05db9d20f582673638f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f85ff14ecb1201f01ba8ba107f90885
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.821779625276395e+26]], [[3.353424750722504e+26]], [[3.168796085442204e+26]], [[2.8879034551481327e+26]], [[3.107271397802924e+26]], [[3.129324111408162e+26]], [[3.0188801342249297e+26]], [[3.0325847737996107e+26]], [[3.3450628416338915e+26]], [[2.8102961581556293e+26]], [[3.273419377000418e+26]], [[3.0883162614625428e+26]], [[3.3082542085092114e+26]], [[3.104119956045371e+26]], [[3.169809549561614e+26]], [[3.25507298321447e+26]], [[3.399893205848823e+26]], [[2.9835197549774766e+26]], [[3.181134005748464e+26]], [[2.822869827851151e+26]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_366ef5a4fa4734e954e72d04b6ee3ad6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 672, 12, 12], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eaf0b509dd9b1ef2f9d4cd21336e7058(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_366ef5a4fa4734e954e72d04b6ee3ad6
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 12, 12], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ebb9deba38bc6d7f056f2f423928fa79(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 28, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b29d3dbf45fb91f322bbc7588a161f63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ebb9deba38bc6d7f056f2f423928fa79
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.3643148126548069e+30]], [[2.4543545498896123e+30]], [[2.4150363532268308e+30]], [[2.122631954079366e+30]], [[2.2961138670041574e+30]], [[2.353872564772158e+30]], [[2.194664892346739e+30]], [[2.4385765567863668e+30]], [[2.12105067910731e+30]], [[2.3861554176276922e+30]], [[2.044368514369154e+30]], [[2.2043763445714306e+30]], [[2.3333761341949567e+30]], [[2.3015955900174724e+30]], [[2.0953262488231852e+30]], [[2.2908567019618357e+30]], [[2.269467479666849e+30]], [[2.3490779649715664e+30]], [[2.0903125312177884e+30]], [[2.1787747713737242e+30]], [[2.1739327212347127e+30]], [[2.1663988466426018e+30]], [[2.345596863073986e+30]], [[2.398099000262575e+30]], [[2.358719752845903e+30]], [[2.2572679069896628e+30]], [[2.3374963045039308e+30]], [[2.104734714014336e+30]]]], dtype='float32').reshape([1, 28, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d7eeb2457870508f97a49ccdfe707eec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ebb9deba38bc6d7f056f2f423928fa79
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.1115171868078946e+34]], [[2.212368200400349e+34]], [[2.1473912036183378e+34]], [[2.260404730508763e+34]], [[2.296651367271031e+34]], [[2.304113917415851e+34]], [[2.1018805663660815e+34]], [[2.0387568896168846e+34]], [[2.342834701140635e+34]], [[2.0004917916205617e+34]], [[2.456818525433852e+34]], [[2.200725126742862e+34]], [[2.2526844412477636e+34]], [[2.224148932990204e+34]], [[2.2247723595939883e+34]], [[2.0508742181034216e+34]], [[2.140541433792964e+34]], [[1.9629482888731382e+34]], [[2.0138078174471388e+34]], [[2.3299175395947156e+34]], [[2.352923417284795e+34]], [[2.3292512802655723e+34]], [[2.145086654441204e+34]], [[2.464766843250088e+34]], [[2.2469542643939195e+34]], [[2.2768368990022293e+34]], [[2.2937280956622626e+34]], [[2.208974511576652e+34]]]], dtype='float32').reshape([1, 28, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4151ba9da213b024c4dd472b68b3b60c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ebb9deba38bc6d7f056f2f423928fa79
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.496062578572521e+38]], [[1.449623162247616e+38]], [[1.4040660307404698e+38]], [[1.4381327715589552e+38]], [[1.4410262601130122e+38]], [[1.5174655913067744e+38]], [[1.4237032568946293e+38]], [[1.3989888365564357e+38]], [[1.3946199041157611e+38]], [[1.4864938447817582e+38]], [[1.3905615967781665e+38]], [[1.3465895355815457e+38]], [[1.377639470795584e+38]], [[1.4136798928926007e+38]], [[1.375850967916734e+38]], [[1.3534947833431569e+38]], [[1.5114691983195028e+38]], [[1.4758988210650507e+38]], [[1.3900932759404181e+38]], [[1.4132580187728447e+38]], [[1.4416011650132277e+38]], [[1.3827135212061295e+38]], [[1.3572852614619353e+38]], [[1.412253938085416e+38]], [[1.418896630054708e+38]], [[1.3987816517423344e+38]], [[1.3919276170649724e+38]], [[1.4126444758823343e+38]]]], dtype='float32').reshape([1, 28, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_fbc77978c5f6f0e2055f03f4b99dda49(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 672, 6, 6], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0454c17e4f4e7802787940fc1d12999b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fbc77978c5f6f0e2055f03f4b99dda49
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 6, 6], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e63ccd8f8948421ad70fd4518ee0b6bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ebb9deba38bc6d7f056f2f423928fa79
    def get_inputs(self):
        return [
            paddle.uniform([1, 28, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_50430e09fdeada4ef8c6191366df4f94(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1152, 6, 6], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7ce3f9055f48e0e422a4e80825471fd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50430e09fdeada4ef8c6191366df4f94
    def get_inputs(self):
        return [
            paddle.uniform([1, 1152, 6, 6], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6805d17f42c4d352974081cacfd8806e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_de1f4ede93b0f394aa640f3d898e3cae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6805d17f42c4d352974081cacfd8806e
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_41d01f866774ba6c371b2411c5f52100(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1280, 6, 6], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_40178c37dcacea518cbfcb6f83c6563f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41d01f866774ba6c371b2411c5f52100
    def get_inputs(self):
        return [
            paddle.uniform([1, 1280, 6, 6], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_40f4200a073923ec2a3ef43b22c01bfc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 112, 112], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_285d500fde5405bf86629bf46b2a54bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40f4200a073923ec2a3ef43b22c01bfc
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 112, 112], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a3da64f7f5990f021b0124c25fcad265(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 112, 112], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_70d77a35828941e4f30bea330f8cb830(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3da64f7f5990f021b0124c25fcad265
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 112, 112], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_04e2928e420437008a197f98d9d4f9f6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 162, 56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bf57f168e6c625d023f6ece6cb342719(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04e2928e420437008a197f98d9d4f9f6
    def get_inputs(self):
        return [
            paddle.uniform([1, 162, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c9e6aee3a43236ad20011d8978b33a37(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 228, 56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d36af4c076a33442dcf3c661ea03141f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9e6aee3a43236ad20011d8978b33a37
    def get_inputs(self):
        return [
            paddle.uniform([1, 228, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6134b1a00aee304c40df2ef4b4425fe4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 300, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d2345d43700b4575b29acc39bdf2a208(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6134b1a00aee304c40df2ef4b4425fe4
    def get_inputs(self):
        return [
            paddle.uniform([1, 300, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c5cd048b470a6dd93d1d5c1b5873c79f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 366, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f6a194e3b7d9a7e38520fa214631fa04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5cd048b470a6dd93d1d5c1b5873c79f
    def get_inputs(self):
        return [
            paddle.uniform([1, 366, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ed562f7044e097c4ac5d0bd95f66247f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 432, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6f1c90b4b223cb991f638be062c7d87c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ed562f7044e097c4ac5d0bd95f66247f
    def get_inputs(self):
        return [
            paddle.uniform([1, 432, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6b165432cf776f255b330b71d404f345(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 504, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ceced35de16de6b5eda434e12a935800(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6b165432cf776f255b330b71d404f345
    def get_inputs(self):
        return [
            paddle.uniform([1, 504, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2a8893f4941ff7d9afa6b44e2189fe87(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 570, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_72e3e8d1cb9d87b5e994ef10e0106b5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a8893f4941ff7d9afa6b44e2189fe87
    def get_inputs(self):
        return [
            paddle.uniform([1, 570, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_43ae63baeea2f9cccf3276595c82b59c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 636, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_43148a4009db2476120c458186db808b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43ae63baeea2f9cccf3276595c82b59c
    def get_inputs(self):
        return [
            paddle.uniform([1, 636, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f4c3b91cd70e06416abcf0ff480b2587(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 702, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d8ccd996afa03e54cfb68ab96f275f50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4c3b91cd70e06416abcf0ff480b2587
    def get_inputs(self):
        return [
            paddle.uniform([1, 702, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b013f7d7c7264e41a7e80ec327e129e2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 768, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_683d4ad058d9291aae7fed795327d295(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b013f7d7c7264e41a7e80ec327e129e2
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4830d7c05595785e2d992b497b0193ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 840, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_379a4e13f3de2b7cbddd6ccb63c5a295(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4830d7c05595785e2d992b497b0193ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 840, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2bd86e7a202faaaa3918c2b406fb0dab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 906, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_776c74c43e938b034b0c33515748db84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bd86e7a202faaaa3918c2b406fb0dab
    def get_inputs(self):
        return [
            paddle.uniform([1, 906, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_cff4f1e7509703145e1b67b575b96bf3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 972, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e5fd5b16952556b1bb97164d6eea2a87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cff4f1e7509703145e1b67b575b96bf3
    def get_inputs(self):
        return [
            paddle.uniform([1, 972, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_cbdf8a41926997b38a692b1dabc664ea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1044, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e9643ffca5ef067664a67be7c6eca293(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbdf8a41926997b38a692b1dabc664ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 1044, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_43c992dda1e50146f05ff6eaf038a5c3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1280, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0eeebe3af6c1f80d2be5c4b2e4721f6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43c992dda1e50146f05ff6eaf038a5c3
    def get_inputs(self):
        return [
            paddle.uniform([1, 1280, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_da95b1327d5d72d400dca0b8837463b7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 320, 320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d995fc2c2085781ffae20b54e6253c87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da95b1327d5d72d400dca0b8837463b7
    def get_inputs(self):
        return [
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

class PrimitiveOp_1d50f3e8e3a8daf81418fc1b7d8fcafa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 320, 320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e4adec966b8e68a1e39b9f9e4be647f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d50f3e8e3a8daf81418fc1b7d8fcafa
    def get_inputs(self):
        return [
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

class PrimitiveOp_87c5ff624678b32608ff367aeb778ec0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, 160, 160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0327a9fa811f70c7fdd5962c9cfa61be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87c5ff624678b32608ff367aeb778ec0
    def get_inputs(self):
        return [
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

class PrimitiveOp_2427b689f1bea3cecd1a3bd025ed6c17(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 24, 160, 160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c20eccb8436bed89b21792bd08c22273(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2427b689f1bea3cecd1a3bd025ed6c17
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 160, 160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c477ff89aea86b3e5038a58ed32ce2cd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 160, 160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7c0d0ea14e047cc96a32e7624068c0d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c477ff89aea86b3e5038a58ed32ce2cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 160, 160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_cfbedc54ac592b5370ed7b4c068339c7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 80, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ccc69dc5146554b1f2506c1e20843ad5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cfbedc54ac592b5370ed7b4c068339c7
    def get_inputs(self):
        return [
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

class PrimitiveOp_845b9c06b06340eaea9f4b50d80ef120(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, 80, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a2ebbe27674abbb11026ac79e30b820d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_845b9c06b06340eaea9f4b50d80ef120
    def get_inputs(self):
        return [
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

class PrimitiveOp_e132e99c4aa1d07ef3fa902aa92ee36e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 80, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4f0fc8e8e24f23ea21fa9a8d0b25d640(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e132e99c4aa1d07ef3fa902aa92ee36e
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 80, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b54937ef3c7e83902191005023bce200(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, 40, 40], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dcc3ea35f514fb8060abefd891ba26c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b54937ef3c7e83902191005023bce200
    def get_inputs(self):
        return [
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

class PrimitiveOp_02d5971b1c2f8d67d98dc903a2317518(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 40, 40], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_82597985dd67217116b097679b9af2bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02d5971b1c2f8d67d98dc903a2317518
    def get_inputs(self):
        return [
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

class PrimitiveOp_af22232e04bc30a6c224b2a2d09c70c7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 40, 40], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3c27c3e2aa59698cba7b42e79f1ef117(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af22232e04bc30a6c224b2a2d09c70c7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_fbd7beab8444ace11dfa1d654b86c029(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 384, 20, 20], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c03c93a908b5760057097962dae5abbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fbd7beab8444ace11dfa1d654b86c029
    def get_inputs(self):
        return [
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

class PrimitiveOp_8d23b79ee053ef6c8084cd2ab03c8ca3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, 20, 20], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4fdd929e2415bafcd4588b871d3c2ef8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d23b79ee053ef6c8084cd2ab03c8ca3
    def get_inputs(self):
        return [
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

class PrimitiveOp_59dcd9551ab05cf152b82fa68af8b078(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, 20, 20], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ddbd033a2bc5fae326d508ac546bfe5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59dcd9551ab05cf152b82fa68af8b078
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 20, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2baf39f918140533c69b3494d5b51fcf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, 56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_46cd651425149752513b68ccbf8e6c67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2baf39f918140533c69b3494d5b51fcf
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1e6ea80e7ca41c32de8ab46e875c3923(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d47f91714db8ea76cca30824e439b7c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e6ea80e7ca41c32de8ab46e875c3923
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_aacbfd2635e3191f2daceb0b4ef2f514(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9d09356f3778aafee4fc35ed44f300cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aacbfd2635e3191f2daceb0b4ef2f514
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[19006174.0]], [[19235500.0]], [[17557304.0]], [[19838796.0]], [[19611992.0]], [[16834778.0]], [[21152790.0]], [[19725676.0]], [[20947212.0]], [[21822048.0]], [[18394660.0]], [[17615450.0]], [[16969952.0]], [[22496512.0]], [[17739876.0]], [[23222528.0]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_658c822848e4a0a114189e2a7849cd2e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_19375d3009ca1f3fe9d7d271ae578cac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_658c822848e4a0a114189e2a7849cd2e
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_650aea4be01ae960f1c04358848e04d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f85ff14ecb1201f01ba8ba107f90885
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[7190799360.0]], [[6596408832.0]], [[7201764864.0]], [[8452954624.0]], [[7108325376.0]], [[6885442560.0]], [[8143262720.0]], [[6375717376.0]], [[8458877952.0]], [[6819305472.0]], [[6888590336.0]], [[7887463424.0]], [[7887997952.0]], [[6318848512.0]], [[7731600384.0]], [[7861496832.0]], [[8238494720.0]], [[6979163136.0]], [[6549292032.0]], [[7918797824.0]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b9d0fc9efd68a2e5f2f5aef40c1f10e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f85ff14ecb1201f01ba8ba107f90885
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1677949403136.0]], [[1673058320384.0]], [[1420855476224.0]], [[1648670015488.0]], [[1598518984704.0]], [[1477845188608.0]], [[1612381159424.0]], [[1935770648576.0]], [[1515699437568.0]], [[1627987247104.0]], [[1952477347840.0]], [[1728554467328.0]], [[1562357923840.0]], [[1374748934144.0]], [[1324509822976.0]], [[1291124998144.0]], [[1614997094400.0]], [[1541365563392.0]], [[1449140158464.0]], [[1271105847296.0]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7ba285f83a042788fdfa7c0097699b23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f85ff14ecb1201f01ba8ba107f90885
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[369381112545280.0]], [[391802150453248.0]], [[337451453251584.0]], [[250776597299200.0]], [[303183452700672.0]], [[335065733136384.0]], [[293844549632000.0]], [[369366650585088.0]], [[396570134577152.0]], [[349206376087552.0]], [[356253645668352.0]], [[322784844578816.0]], [[397283770236928.0]], [[356633817382912.0]], [[318540477366272.0]], [[331936547471360.0]], [[407114782605312.0]], [[294153820831744.0]], [[326793827450880.0]], [[281171443318784.0]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f75ed0e773b4981be911642e4c02622c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a41fa467fc39e26287877ba9e43e9e85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f75ed0e773b4981be911642e4c02622c
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cb7b1b9cd0d8f2b4db50fcb5d77fb4e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff33d5afbedb70bb059ced37dbdb332d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.5312278089826304e+17]], [[4.292216455346258e+17]], [[4.313428096230359e+17]], [[2.614324087446569e+17]], [[4.239283904000819e+17]], [[2.525378578921554e+17]], [[3.968344997463327e+17]], [[3.8454491468529664e+17]], [[3.526352661811036e+17]], [[2.9179368717890355e+17]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c228375dcbdafc61b7dcced4e0ada903(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 480, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7bdfff3a6a50f1a17ab6f76be967bbfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c228375dcbdafc61b7dcced4e0ada903
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b0bce4578b99bbaf44728f5c3012a28c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f85ff14ecb1201f01ba8ba107f90885
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.8778166785037514e+21]], [[1.665290404713254e+21]], [[1.4771042573213457e+21]], [[1.4230918833028496e+21]], [[1.674811295800492e+21]], [[1.9973043542297967e+21]], [[1.8161486073311904e+21]], [[1.8495689757409768e+21]], [[1.274819591295954e+21]], [[2.1021084655832657e+21]], [[1.9342070938629875e+21]], [[1.2905325096583614e+21]], [[1.8269430319505557e+21]], [[1.3110839828704248e+21]], [[1.5834585921090486e+21]], [[2.1872249504281961e+21]], [[1.4959183262396627e+21]], [[1.9333789944815047e+21]], [[1.955100840646731e+21]], [[1.7115378692867217e+21]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9abfb30e385d47e18a1d5820dbe2f45a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f85ff14ecb1201f01ba8ba107f90885
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.390321905160308e+24]], [[3.0724640404135854e+24]], [[3.2631428457567503e+24]], [[3.384238803071626e+24]], [[3.7169771374479327e+24]], [[2.550668196559967e+24]], [[3.1074592433036693e+24]], [[3.13905131014253e+24]], [[3.1042699741915507e+24]], [[2.766700034019813e+24]], [[3.500168519124358e+24]], [[3.197247905391322e+24]], [[2.7528551761317414e+24]], [[3.3659693206792496e+24]], [[2.6314568651522827e+24]], [[3.3629065847022615e+24]], [[3.0457563255289917e+24]], [[3.685714229698637e+24]], [[3.2631978977585953e+24]], [[3.40785092371635e+24]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_988efc70da245751ae73085c74f00832(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f85ff14ecb1201f01ba8ba107f90885
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.191811218787006e+27]], [[7.095622454217924e+27]], [[7.310751039936671e+27]], [[6.747469527043222e+27]], [[5.513372903779087e+27]], [[6.093075855820906e+27]], [[7.568960594487018e+27]], [[8.52792093083877e+27]], [[8.461526228977054e+27]], [[6.218124710583104e+27]], [[7.604449178605783e+27]], [[5.562950668299494e+27]], [[6.126870290963941e+27]], [[5.684747583442426e+27]], [[5.406959097454102e+27]], [[5.535262843314619e+27]], [[5.514021638874671e+27]], [[8.292876354774331e+27]], [[9.46577937256182e+27]], [[9.092308067785022e+27]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_acda317bf8f0c83d825cd2096de6e66c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 40, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e47f67e278a7126efd75b8434ff15b6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_acda317bf8f0c83d825cd2096de6e66c
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d1fe3058386f1e38d1d7a6b6e66c621b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 360, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3f0961295a564b75074eebfa313393f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1fe3058386f1e38d1d7a6b6e66c621b
    def get_inputs(self):
        return [
            paddle.uniform([1, 360, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ae8cb7dc893aa2a83f250ad4fcd1a043(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 60, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_47884db3a642357d62cf859c633a5149(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae8cb7dc893aa2a83f250ad4fcd1a043
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d3cd38e84d2f8529181c6b46d71d3a6e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 720, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5b804b7c4f58b40930a211a6b406f6bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3cd38e84d2f8529181c6b46d71d3a6e
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1df830ca8155d2262d6656b51f0144d2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 720, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d8489f35711798390a90ed316c7ce962(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1df830ca8155d2262d6656b51f0144d2
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_275ff2047afc54ceb8f38bf4994a181d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1200, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b552ddfba6eb79fb2e81a30c47d773a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_275ff2047afc54ceb8f38bf4994a181d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1200, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5afd4a66990530436ac42996497c5a64(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 100, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9824554a24de892014c97e9ef81b7cec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5afd4a66990530436ac42996497c5a64
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
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2e3bb683e6ccc14fddac1dbf09004b7f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, 56, 56], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f659e7b1bfd59a45ff068b940db695da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2e3bb683e6ccc14fddac1dbf09004b7f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 56, 56], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_37b4b67af87b4b9580d2713d21293397(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, 28, 28], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_837d93d8813247262f1717d7a284dc4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37b4b67af87b4b9580d2713d21293397
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 28, 28], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d352af92189b382a88389c75c594c6db(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e67f2d41127652d3aed74eefa5ac8fd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d352af92189b382a88389c75c594c6db
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ae9fa94ee4d35468e86e2f27340af165(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, 28, 28], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_16ba0c936b0990075560bcd646d032aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae9fa94ee4d35468e86e2f27340af165
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 28, 28], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e1909e352c3b6a762606ec8811cfe583(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 20, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7865ef717e856719a25c7b9d79ba19b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1909e352c3b6a762606ec8811cfe583
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ec9acf295dcacfa02c1fdbb1bf76cad3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, 14, 14], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d69f55d5a8fb19734207919a708034ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec9acf295dcacfa02c1fdbb1bf76cad3
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9a9ac243f3ec436b52c11bcb7067ffaf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 10, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6844f5265e0172fc0dbf513415e564db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a9ac243f3ec436b52c11bcb7067ffaf
    def get_inputs(self):
        return [
            paddle.uniform([1, 10, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_18fcfb6ef3a3b74a196942d57f8efc69(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 480, 14, 14], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d0f11a199316e565b3aab0e22a1816cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_18fcfb6ef3a3b74a196942d57f8efc69
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_fc083c25b3600ce2e7b58b5b17b84064(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 40, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_840a1d6aa5a1ba83a544ac51fcc9c662(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc083c25b3600ce2e7b58b5b17b84064
    def get_inputs(self):
        return [
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

class PrimitiveOp_081a72bb4186f6278afe9eb95474b4ee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 360, 14, 14], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1d241cd6a2fa187d526ae6a2bc9d92f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081a72bb4186f6278afe9eb95474b4ee
    def get_inputs(self):
        return [
            paddle.uniform([1, 360, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_10374c9c4941da6564584e11fb23ad1c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 60, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c459b2b43fb4c6a8b6c19242e96a9131(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10374c9c4941da6564584e11fb23ad1c
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b27125bd890f82a6dfc2016f696db45d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 720, 14, 14], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bbd7dc02ba6fb464962ecb91f525863b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b27125bd890f82a6dfc2016f696db45d
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b578e75678d0bb8bc513c8a5edc7ae9d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 720, 7, 7], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d72eb845f865066a08c0fa42dd3770e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b578e75678d0bb8bc513c8a5edc7ae9d
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5a244ba2a55fe16221f6029331507315(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1200, 7, 7], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5940b1fee86bb03a52a6d67a96ad0bcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a244ba2a55fe16221f6029331507315
    def get_inputs(self):
        return [
            paddle.uniform([1, 1200, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_bb9216b68d2aa6e73c4862f2c0e499da(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 100, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_78fd66cdca87fcf0c1ee6d447a9fd17e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb9216b68d2aa6e73c4862f2c0e499da
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_3dc1690e533876ca19cdfde4781b1365(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 96, 96], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f23beaa623ae9ba8d9993d71cfe903f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3dc1690e533876ca19cdfde4781b1365
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 96, 96], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a017166d18a8d46098cd9fd671652a44(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7dc6e375a648bf0ad192da2577a4124f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a017166d18a8d46098cd9fd671652a44
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.203125]], [[2.28515625]], [[2.30078125]], [[3.12890625]], [[2.748046875]], [[2.375]], [[2.8984375]], [[3.3671875]]]], dtype='float16').reshape([1, 8, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_cf1f6ce76d1420c61e9792b569a92c04(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 96, 96], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a605c46bec7168bd56a6d745acf01253(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf1f6ce76d1420c61e9792b569a92c04
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 96, 96], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8b8473edeeaadd655a47801c1a239bec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 48, 48], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6637fe18677d00757f13332d2b4a1b5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b8473edeeaadd655a47801c1a239bec
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 48, 48], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b4dfc9c024095308c1ae3e63ee7fbddb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c8e457ac3a34f1ea92c14106fc508e97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4dfc9c024095308c1ae3e63ee7fbddb
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[79.875]], [[79.9375]], [[77.75]], [[65.75]]]], dtype='float16').reshape([1, 4, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1823451a12818f79746b8556a29ceb3c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, 48, 48], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c252ee679cf62146b214c678fb07723d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1823451a12818f79746b8556a29ceb3c
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 48, 48], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1afc2dc807288bbd8add4e189dadfb7f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 6, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_27113c63f4f1e64d16c3385d2ce1212c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1afc2dc807288bbd8add4e189dadfb7f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[12784.0]], [[14912.0]], [[12584.0]], [[12112.0]], [[14512.0]], [[12680.0]]]], dtype='float16').reshape([1, 6, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8fe23ba84c21455bf7f8ffb82917ea5d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, 24, 24], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_90306acb8e11c86b88d7f63913174544(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8fe23ba84c21455bf7f8ffb82917ea5d
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 24, 24], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_89139e4074a2b74ca4b32b6de5ccd052(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1afc2dc807288bbd8add4e189dadfb7f
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_89e15e710e38e63b6aec83dc5a970d14(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, 24, 24], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8291fbc92880c818bdb0d49d7e146922(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89e15e710e38e63b6aec83dc5a970d14
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 24, 24], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_85367fcc6fee992391e8ac9fc972716d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, 12, 12], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_edd798efe1ad68536f8d56e3602dc6c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85367fcc6fee992391e8ac9fc972716d
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 12, 12], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a6476c6ddd439b173d9aa06762d0d1ad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 480, 12, 12], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e0db06bbc18244740b894c6cc44f975f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6476c6ddd439b173d9aa06762d0d1ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 12, 12], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_18b613424751ba165a1c96cd8b69c826(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 672, 12, 12], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e72068bc0fbd25af157cd685736fc4c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_18b613424751ba165a1c96cd8b69c826
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 12, 12], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_49f2efb639a1b925bbc25b8290798a2c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 28, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_37034a2b69e185284fb8cb76371616d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_49f2efb639a1b925bbc25b8290798a2c
    def get_inputs(self):
        return [
            paddle.uniform([1, 28, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_7df7b4344cb169e4c95073d50e7cbf6e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 672, 6, 6], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8e95a5b43b6a9621ee989f054c62ef47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7df7b4344cb169e4c95073d50e7cbf6e
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 6, 6], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_75ad211c5894b03d736a81b791ee7199(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1152, 6, 6], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_36bc4b21b99a48e59bb1604b53c703c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75ad211c5894b03d736a81b791ee7199
    def get_inputs(self):
        return [
            paddle.uniform([1, 1152, 6, 6], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ca6af9a970a1cdede5fd8d0a47175acc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_384022f6aa2035e9c307cd9f914f12ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca6af9a970a1cdede5fd8d0a47175acc
    def get_inputs(self):
        return [
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

class PrimitiveOp_5e91723fa5c596957f17a6b1f54ee101(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1280, 6, 6], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_abfde652a068f8edfb0c9a7c9c251435(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e91723fa5c596957f17a6b1f54ee101
    def get_inputs(self):
        return [
            paddle.uniform([1, 1280, 6, 6], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_aa3bf7ef09e34a67b0d0f1be04108b50(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 112, 112], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d7083222986325faf0f34e5105e97614(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa3bf7ef09e34a67b0d0f1be04108b50
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 112, 112], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_73ae3bc953ccc3aaa7fbd882482dce6a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 112, 112], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3f29f8c9b15a678626ac3d501462690c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73ae3bc953ccc3aaa7fbd882482dce6a
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 112, 112], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9bbea6faafdf76b3b2011c2c75aa9ac0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 162, 56, 56], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a8034f66faa0735332771866c5fe9ab1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9bbea6faafdf76b3b2011c2c75aa9ac0
    def get_inputs(self):
        return [
            paddle.uniform([1, 162, 56, 56], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1249b5d423a5387298ac8b2b4d0c7acc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 228, 56, 56], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8f4790cba8803d2bcec2b637efc14844(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1249b5d423a5387298ac8b2b4d0c7acc
    def get_inputs(self):
        return [
            paddle.uniform([1, 228, 56, 56], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5cb9484bb86b90704334db7557a2b4dd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 300, 28, 28], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7009ec1d56b9713db8b8a54ad4a98f41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5cb9484bb86b90704334db7557a2b4dd
    def get_inputs(self):
        return [
            paddle.uniform([1, 300, 28, 28], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2cc7fcc0f0022ba2ade0301aa5a1b337(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 366, 28, 28], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5b33c08d552008447cd7717f8b12a38c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cc7fcc0f0022ba2ade0301aa5a1b337
    def get_inputs(self):
        return [
            paddle.uniform([1, 366, 28, 28], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_86c87ffae2fc445b840f8fdb054fac9d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 432, 14, 14], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fbb4534ad3273f1352339a7d45e48cbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86c87ffae2fc445b840f8fdb054fac9d
    def get_inputs(self):
        return [
            paddle.uniform([1, 432, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5a40fcf1d915d4d6370418640ce67267(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 504, 14, 14], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1b2b63978ccd5941fef449b84ed62e10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a40fcf1d915d4d6370418640ce67267
    def get_inputs(self):
        return [
            paddle.uniform([1, 504, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2611edb453a5cac5a04c794f519b6cde(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 570, 14, 14], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eb08b6592959a1f717e92601cc0b2e8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2611edb453a5cac5a04c794f519b6cde
    def get_inputs(self):
        return [
            paddle.uniform([1, 570, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d442a37eb726e5d2f0df50e40b51cae4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 636, 14, 14], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fae29476ca36b70c9f8aa957801e4b30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d442a37eb726e5d2f0df50e40b51cae4
    def get_inputs(self):
        return [
            paddle.uniform([1, 636, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0e7359142b465b76c672037b66ca9fcc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 702, 14, 14], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0c907230318ad3ce097d14bf3ecd3335(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e7359142b465b76c672037b66ca9fcc
    def get_inputs(self):
        return [
            paddle.uniform([1, 702, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_dcd1037552173753bb2edc6cb5694eff(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 768, 14, 14], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c8e28c9a431f44130c0d00c4dc43f0c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dcd1037552173753bb2edc6cb5694eff
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a72b0d966f09d8c40e61223cbdeede84(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 840, 7, 7], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e906e317e5bc126f62d04afa863f1545(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a72b0d966f09d8c40e61223cbdeede84
    def get_inputs(self):
        return [
            paddle.uniform([1, 840, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_38f117cc1566c69a250fea18ff27f7b3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 906, 7, 7], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e55de75c2e15c39b1b914244e62523d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38f117cc1566c69a250fea18ff27f7b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 906, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_83bbce45b225a7a2a66e3ff2e778e6fe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 972, 7, 7], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a8d96016ddf1b17593e5bdc4164a0a79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_83bbce45b225a7a2a66e3ff2e778e6fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 972, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ac3eac5ed5fbc3ed098b68b8f353650c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1044, 7, 7], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b225c746c5d611cba307f6e39375c167(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac3eac5ed5fbc3ed098b68b8f353650c
    def get_inputs(self):
        return [
            paddle.uniform([1, 1044, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4b6400392149d6fb405cb9bb9539fc8f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1280, 7, 7], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b292993fa19500f034e5f40752db60fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b6400392149d6fb405cb9bb9539fc8f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1280, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a02adbc5e06ba0c5471f233f792b6d95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29007ee7b2e66f3b636fcfe61bbce4a2
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.922877788543701]], [[3.5791587829589844]], [[4.144195556640625]], [[4.144674301147461]], [[3.3976290225982666]], [[3.6537511348724365]], [[3.263800859451294]], [[3.7594494819641113]]]], dtype='float32').reshape([1, 8, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a7532207870f6a96f69dc32ec3988d86(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e33bf8458bc23297cd797190848b89a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7532207870f6a96f69dc32ec3988d86
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_49ce17b3ae5dac563cd3452ceaa888a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10b285d1685c7e1ec0629a08f4f1c2b5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[380.4045715332031]], [[292.51519775390625]], [[319.36065673828125]], [[269.7727355957031]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_7e0a868c572be307a632e284b2ab8672(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, 56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e97baec600c5ad0e8d792d6056390159(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e0a868c572be307a632e284b2ab8672
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6d3b67fb9d823ea5312763c37371cc74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8d07bd6ef969a87be053df3f69b43d8
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[29749.802734375]], [[34152.0859375]], [[29265.646484375]], [[27124.87890625]], [[30676.791015625]], [[25434.556640625]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f3acdd14a5031f31eda60d1272074fec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3cef37a4834c0b5117a96d44a98849fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3acdd14a5031f31eda60d1272074fec
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dd47ab6b41e035cde5203e46c78f9430(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8d07bd6ef969a87be053df3f69b43d8
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[22595958.0]], [[23814368.0]], [[22048644.0]], [[24586742.0]], [[19240024.0]], [[20879972.0]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_70d81a99c2b4a183d062730640cad0b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff33d5afbedb70bb059ced37dbdb332d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[17820053504.0]], [[17573085184.0]], [[12539602944.0]], [[16684223488.0]], [[15716587520.0]], [[13704207360.0]], [[16981073920.0]], [[16666883072.0]], [[13481972736.0]], [[13344173056.0]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8bf3ed83bbcda281793a1c9d46f64e74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff33d5afbedb70bb059ced37dbdb332d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[11998731436032.0]], [[12011637309440.0]], [[11930318143488.0]], [[12494372339712.0]], [[11370160455680.0]], [[13144015503360.0]], [[11300576952320.0]], [[10547793756160.0]], [[12579927752704.0]], [[12332767903744.0]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_aff0ce79a06006fdb8d3559fde66bc86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f85ff14ecb1201f01ba8ba107f90885
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.3558145527840768e+16]], [[1.2930531620552704e+16]], [[1.2398914527297536e+16]], [[1.4013161879371776e+16]], [[1.2977771965841408e+16]], [[1.4001563320188928e+16]], [[1.4354497962770432e+16]], [[1.3874335651463168e+16]], [[1.2451473115840512e+16]], [[1.3568060459843584e+16]], [[1.4111631520825344e+16]], [[1.361962583719936e+16]], [[1.3106773456060416e+16]], [[1.3741105765941248e+16]], [[1.3777262948122624e+16]], [[1.3786289895636992e+16]], [[1.2567663792357376e+16]], [[1.407670699425792e+16]], [[1.3706006219456512e+16]], [[1.372359088930816e+16]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_36a4dbdf0541bf8eeacec87696e1d692(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f85ff14ecb1201f01ba8ba107f90885
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.4546745543662502e+19]], [[1.515836977733383e+19]], [[1.4809108809259155e+19]], [[1.5719047040220529e+19]], [[1.6000018440605663e+19]], [[1.428553346674013e+19]], [[1.587737341608539e+19]], [[1.5447489658392412e+19]], [[1.5669181988877631e+19]], [[1.5068736489925902e+19]], [[1.352543118285393e+19]], [[1.4825126494652596e+19]], [[1.436059052849863e+19]], [[1.5100369439457018e+19]], [[1.4507147731899777e+19]], [[1.5027874239791235e+19]], [[1.4982748083564052e+19]], [[1.5746264351054496e+19]], [[1.5435203715463643e+19]], [[1.6264833617132257e+19]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2c5e72ca01b0baecdc319a72d83ae851(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f85ff14ecb1201f01ba8ba107f90885
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.286318036848084e+22]], [[5.697436784951967e+22]], [[5.176352293666691e+22]], [[5.826673430938879e+22]], [[5.557613977680958e+22]], [[5.964559690370118e+22]], [[5.316860548800986e+22]], [[5.359184027019125e+22]], [[5.915843352480926e+22]], [[5.509547959577958e+22]], [[5.105060761925379e+22]], [[5.33430929519727e+22]], [[5.102122163168519e+22]], [[6.224474584584664e+22]], [[5.946090878658235e+22]], [[5.608352431802839e+22]], [[6.118477863754872e+22]], [[4.8608278519733e+22]], [[5.67184778222921e+22]], [[5.571804820106802e+22]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_3b031f294275ad91c6827d1eb6f8bd78(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 672, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7232aada3855960d4344a45dd474de72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b031f294275ad91c6827d1eb6f8bd78
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_966e6f59ef27b856037dbace314d84dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ebb9deba38bc6d7f056f2f423928fa79
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.6257573538966114e+26]], [[3.5318741256729925e+26]], [[3.5369484560327885e+26]], [[3.3819795726788434e+26]], [[3.5437685862517204e+26]], [[3.625693897096998e+26]], [[3.411988735937954e+26]], [[3.440398566485874e+26]], [[3.682524626239282e+26]], [[3.391286692933793e+26]], [[3.612572359102487e+26]], [[3.3940304616473164e+26]], [[3.470999870229751e+26]], [[3.3713424421761796e+26]], [[3.4559281424517673e+26]], [[3.416584188821597e+26]], [[3.591479614058844e+26]], [[3.704010286931695e+26]], [[3.415231304611231e+26]], [[3.477954661680421e+26]], [[3.8338384717224625e+26]], [[3.627645931554878e+26]], [[3.3974958669890035e+26]], [[3.3403939707088356e+26]], [[3.466741623827776e+26]], [[3.727230310501917e+26]], [[3.4337994282609454e+26]], [[3.5828771594275106e+26]]]], dtype='float32').reshape([1, 28, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b48e45875e2e6e173f382c191001d488(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ebb9deba38bc6d7f056f2f423928fa79
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.3777476407837274e+30]], [[2.2989069389946495e+30]], [[2.2962497200431365e+30]], [[2.4124232600677338e+30]], [[2.429969458297893e+30]], [[2.2865320720736193e+30]], [[2.3708974137426085e+30]], [[2.3604315918064772e+30]], [[2.1471603033807095e+30]], [[2.4524716479255625e+30]], [[2.4047444654929965e+30]], [[2.261522570296069e+30]], [[2.1575146019099813e+30]], [[2.3544863968570674e+30]], [[2.3048137505492866e+30]], [[2.4102824035569237e+30]], [[2.3344611451180608e+30]], [[2.3898656109680895e+30]], [[2.240436939497533e+30]], [[2.4122933005421252e+30]], [[2.216803044186977e+30]], [[2.3989917919803603e+30]], [[2.3858102693061922e+30]], [[2.2991520487045764e+30]], [[2.400677034572903e+30]], [[2.2883199222451018e+30]], [[2.288468166773732e+30]], [[2.3874943029729154e+30]]]], dtype='float32').reshape([1, 28, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ea19380a1a88097bcfcef18bb98f25c3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 672, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7d3cb2652a0aab1b9dfb3a652932fcd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea19380a1a88097bcfcef18bb98f25c3
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4ae39ee2ec3d78acfd532d4ca08e450d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ebb9deba38bc6d7f056f2f423928fa79
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.056422045350911e+34]], [[5.55795067417066e+34]], [[5.133265440173496e+34]], [[5.370417613499397e+34]], [[5.555486183140451e+34]], [[4.787142356949461e+34]], [[5.364995931303342e+34]], [[5.267108050988922e+34]], [[5.3585413119385085e+34]], [[5.582729281996997e+34]], [[4.816064102499278e+34]], [[5.286069330982648e+34]], [[5.021515602475174e+34]], [[5.091482983143552e+34]], [[5.080955541049469e+34]], [[5.248341870345387e+34]], [[5.218366885410146e+34]], [[5.089253700720807e+34]], [[5.390759444224934e+34]], [[4.960452476921367e+34]], [[4.998571126611043e+34]], [[5.309133639442582e+34]], [[4.96736018234058e+34]], [[5.591924205432793e+34]], [[5.434502303101091e+34]], [[5.347184450018104e+34]], [[5.750695957583331e+34]], [[5.187593676737575e+34]]]], dtype='float32').reshape([1, 28, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1844f5acfa6f3aeb58bc5253857258b4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1152, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_87d42ae023ef31cfbda41b642a011ba5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1844f5acfa6f3aeb58bc5253857258b4
    def get_inputs(self):
        return [
            paddle.uniform([1, 1152, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4a8f4c42a9830f9a6f3c36d1a81d3726(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a017166d18a8d46098cd9fd671652a44
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.99609375]], [[6.70703125]], [[6.203125]], [[5.5859375]], [[6.6796875]], [[6.1328125]], [[4.98046875]], [[5.2734375]]]], dtype='float16').reshape([1, 8, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2f2ade257627fdddb333c7239a36d060(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 56, 56], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5f476d07592221575deb75729af8582e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f2ade257627fdddb333c7239a36d060
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 56, 56], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_202271fdbefc998a107398d0de2a1ac2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4dfc9c024095308c1ae3e63ee7fbddb
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1193.0]], [[1526.0]], [[1433.0]], [[1565.0]]]], dtype='float16').reshape([1, 4, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_72b67a94e27adde99ffab51d31fb929c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, 56, 56], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_772d3ef812ab4aaa6639ab4e80630f10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72b67a94e27adde99ffab51d31fb929c
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 56, 56], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_60042b0175ca2892a7c9c0f316421c14(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, 28, 28], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8b69a3a3ed27df2c7700d98c9ab9c2c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60042b0175ca2892a7c9c0f316421c14
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 28, 28], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_135a20185eb5dfca5393a18b92112a33(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 672, 14, 14], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_172ff2a7600ba5d43f0daa06b08a8804(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_135a20185eb5dfca5393a18b92112a33
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6a2b8e83b1279960ef1a750b5067862e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 672, 7, 7], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_116dff2c31cea264c00283ef0829f686(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a2b8e83b1279960ef1a750b5067862e
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d679dd56903a9c0bd61452eac78ccd24(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1152, 7, 7], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3e20cbe5fdb61ffb9a004ae658aa69c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d679dd56903a9c0bd61452eac78ccd24
    def get_inputs(self):
        return [
            paddle.uniform([1, 1152, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_826aad6eea83e34f72af7c9af1cc0a59(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 320, 320], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a7c222ebd8ee39f5e40a32ffcf3bd4e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_826aad6eea83e34f72af7c9af1cc0a59
    def get_inputs(self):
        return [
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

class PrimitiveOp_0c7f25aae5380cd99500ed9661aec94b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 320, 320], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0a23876ce00aa1964f7007b3cf41f8ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c7f25aae5380cd99500ed9661aec94b
    def get_inputs(self):
        return [
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

class PrimitiveOp_f5f8297eae2b837ffe74d20f24216d2a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, 160, 160], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c06c6e473b3d7b570eca587ff0a7da0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5f8297eae2b837ffe74d20f24216d2a
    def get_inputs(self):
        return [
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

class PrimitiveOp_c9b065d4dda3f0656cfac7f62f8a4018(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 24, 160, 160], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fdbdb8bbb7a91521e2191e63f6c254c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9b065d4dda3f0656cfac7f62f8a4018
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 160, 160], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0a048a9d0b01e19683a755a063e940c1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 160, 160], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_718991193deca6edc47e5bb6cbea76d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a048a9d0b01e19683a755a063e940c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 160, 160], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_639512479f39e970c08a355e30a94987(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 80, 80], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_afaf2e0be6805efa41c20e57bbc1b649(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_639512479f39e970c08a355e30a94987
    def get_inputs(self):
        return [
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

class PrimitiveOp_dd69da155ea481490d1d920111b805ea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, 80, 80], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4c11bb29bdbad0bee730caa370e8d56a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd69da155ea481490d1d920111b805ea
    def get_inputs(self):
        return [
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

class PrimitiveOp_6bd0577118b4f850fdcfc9f1c87ba8d5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 80, 80], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_479844d2eab09613e773eb773167fea5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bd0577118b4f850fdcfc9f1c87ba8d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 80, 80], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8f3b225abe3644db4dff03829dea62df(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, 40, 40], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fe4beb95b9a60dce366709c247830a4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f3b225abe3644db4dff03829dea62df
    def get_inputs(self):
        return [
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

class PrimitiveOp_37553785549b288b6902ded01c6ecd18(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 40, 40], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1a9dd60ddeb0e46f21cda9c27a9ebc13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37553785549b288b6902ded01c6ecd18
    def get_inputs(self):
        return [
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

class PrimitiveOp_10b576186e33c7ee4912df9c8580b3e2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 40, 40], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8dabd859ebb660ef173e7d68a7d8b759(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10b576186e33c7ee4912df9c8580b3e2
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_60bc5851a9ec1dff038511f36a0e0715(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 384, 20, 20], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_30e44eecb949839ec44bdf29f803ed9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60bc5851a9ec1dff038511f36a0e0715
    def get_inputs(self):
        return [
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

class PrimitiveOp_4134c34347f20da852c7c9cefdf0bac2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, 20, 20], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b273bd38a66c3fb7cbabacbb6ac3ad98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4134c34347f20da852c7c9cefdf0bac2
    def get_inputs(self):
        return [
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

class PrimitiveOp_c1cd154dcb474e82e2332591e88b9552(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, 20, 20], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_79d08719bc65356f580fe162b09e07ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1cd154dcb474e82e2332591e88b9552
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 20, 20], dtype='float16', min=0, max=0.5),
        ]


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