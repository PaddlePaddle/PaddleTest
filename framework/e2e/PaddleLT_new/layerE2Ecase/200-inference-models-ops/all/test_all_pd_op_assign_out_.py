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
class PrimitiveOp_5515fd315ea737c77d6c99f7d91ae548(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.assign_out_(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5ae0bddcb60d56977e048bd3ef765712(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5515fd315ea737c77d6c99f7d91ae548
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 501, 30], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1fcd9a8bc34100fbec68cf8187c1064f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.assign_out_(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_80d6b8ad5eb62d94ce32ada0148da24a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fcd9a8bc34100fbec68cf8187c1064f
    def get_inputs(self):
        return [
            paddle.to_tensor([1], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4674e5d6af96838b833215c3d8fb5cd1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.assign_out_(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7c1fa3dd2ab82a081c714b30ea744d88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4674e5d6af96838b833215c3d8fb5cd1
    def get_inputs(self):
        return [
            paddle.to_tensor([[16.59449005126953, 18.070903778076172, 16.7813720703125, 17.00531005859375, 16.58167266845703, 16.619869232177734, 15.407929420471191, 15.935876846313477, 16.032629013061523, 16.307723999023438, 16.820934295654297, 15.481019973754883, 16.534021377563477, 16.698375701904297, 17.18429183959961, 17.315828323364258, 15.136454582214355, 17.12547492980957, 17.039785385131836, 16.16747283935547, 16.435935974121094, 15.712438583374023, 17.211814880371094, 16.4127197265625, 17.575082778930664, 16.354190826416016, 17.133848190307617, 16.645544052124023, 16.288793563842773, 16.682811737060547]], dtype='float32').reshape([1, 30]),
            paddle.to_tensor([1.77113e+27], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_aa1438167b65231afd37e38141a76c82(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.assign_out_(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='int64'),
            paddle.static.InputSpec(shape=[], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f26e9dd7f6a64d1c4763b6c7e79f0434(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa1438167b65231afd37e38141a76c82
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int64').reshape([]),
            paddle.to_tensor(0, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5d991f1c41ab24a591935c5cf00cf859(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4674e5d6af96838b833215c3d8fb5cd1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_60f3b98d337ff5e869da142c59f0b69f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4674e5d6af96838b833215c3d8fb5cd1
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.9999996423721313, 0.9999994039535522, 0.9999997615814209, 0.9999998807907104]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([1.77113e+27], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b388e3a8cf871c93300c57ce08a9f005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa1438167b65231afd37e38141a76c82
    def get_inputs(self):
        return [
            paddle.to_tensor(0, dtype='int64').reshape([]),
            paddle.to_tensor([1.77113e+27], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c9677788f03c4c6bbdd4b93a26c101dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5515fd315ea737c77d6c99f7d91ae548
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 501, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_262d424fd1205a1654f6b2d53424620b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.assign_out_(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='bool'),
            paddle.static.InputSpec(shape=[], dtype='bool'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9fe25436599d4fe533e911290004882e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262d424fd1205a1654f6b2d53424620b
    def get_inputs(self):
        return [
            paddle.to_tensor(True, dtype='bool').reshape([]),
            paddle.to_tensor(True, dtype='bool').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d1002bca3cc1509c1cbf1985e6aeac54(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.assign_out_(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ead01070c02bb3622e1acc4cf8814a1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1002bca3cc1509c1cbf1985e6aeac54
    def get_inputs(self):
        return [
            paddle.to_tensor(0.0, dtype='float32').reshape([]),
            paddle.to_tensor([1.77113e+27], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_70f1e8c9cff9775d8a2f52b797c5a830(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.assign_out_(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1620b9d1c17ceb11019aee9df55701a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70f1e8c9cff9775d8a2f52b797c5a830
    def get_inputs(self):
        return [
            paddle.to_tensor(0, dtype='int32').reshape([]),
            paddle.to_tensor([1.77113e+27], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_116285ea819bbe468a1fedd934b368c8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.assign_out_(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8e176e89ff36c1cff3c6de436f37c564(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_116285ea819bbe468a1fedd934b368c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_91cc693a9958a6126154f07391a8eeec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262d424fd1205a1654f6b2d53424620b
    def get_inputs(self):
        return [
            paddle.to_tensor(False, dtype='bool').reshape([]),
            paddle.to_tensor(True, dtype='bool').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f0fa1a1e00e6e3a49c6d75d103b210df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5515fd315ea737c77d6c99f7d91ae548
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.77113e+27], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_65ff15e8005e1a7a12fd80c54fbae2df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_262d424fd1205a1654f6b2d53424620b
    def get_inputs(self):
        return [
            paddle.to_tensor(False, dtype='bool').reshape([]),
            paddle.to_tensor(False, dtype='bool').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7444a619bb7519b7deb4094ea867a508(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa1438167b65231afd37e38141a76c82
    def get_inputs(self):
        return [
            paddle.to_tensor(2, dtype='int64').reshape([]),
            paddle.to_tensor(1, dtype='int64').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a37393250aba543f01d6f6284978207c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.assign_out_(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b2b85e683f656796e3d5a5503a7a48fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a37393250aba543f01d6f6284978207c
    def get_inputs(self):
        return [
            paddle.to_tensor([[2, 0]], dtype='int64').reshape([1, 2]),
            paddle.to_tensor([[2]], dtype='int64').reshape([1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b578374fa118d5c4b7e703b0f8a04b69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4674e5d6af96838b833215c3d8fb5cd1
    def get_inputs(self):
        return [
            paddle.uniform([1, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.77113e+27], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5b8cc18526464411dd8044e171d4d8cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4674e5d6af96838b833215c3d8fb5cd1
    def get_inputs(self):
        return [
            paddle.uniform([1, 99], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.77113e+27], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_3e1fe63f74b41f18eedcf35529ea2d6c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.assign_out_(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8f9c985c2c0a065468d83c1d02a278e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e1fe63f74b41f18eedcf35529ea2d6c
    def get_inputs(self):
        return [
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1.77113e+27], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d09cc67de54d383b7b6cae57e5372b12(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.assign_out_(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e465288350a96b3d8582ececdaeb9b1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d09cc67de54d383b7b6cae57e5372b12
    def get_inputs(self):
        return [
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.77113e+27], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_69da48ad08a2c2d39d89d0ae7fed39df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_116285ea819bbe468a1fedd934b368c8
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.0]]]], dtype='float32').reshape([1, 1, 1, 1]),
            paddle.to_tensor([1.77113e+27], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9f0e1d1209153467cfb7a7a35ae9f8d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4674e5d6af96838b833215c3d8fb5cd1
    def get_inputs(self):
        return [
            paddle.uniform([1, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1.0]], dtype='float32').reshape([1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c08b56b61e487e0c1a5e5dbd767f54d1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.assign_out_(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a462646fac763962a8a58a8ee92481e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c08b56b61e487e0c1a5e5dbd767f54d1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 256], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f4b07553c3facaf59308f7c9ad15b707(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4674e5d6af96838b833215c3d8fb5cd1
    def get_inputs(self):
        return [
            paddle.to_tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype='float32').reshape([1, 30]),
            paddle.to_tensor([1.77113e+27], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_bf7ab9538118f29dbdd784fc9f71f720(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.assign_out_(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f7cf30d4b57af11580bdc9bb9942e9e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf7ab9538118f29dbdd784fc9f71f720
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 256], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 501, 256], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f6add37c6337a1253dbbc2ab8fdc510e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fcd9a8bc34100fbec68cf8187c1064f
    def get_inputs(self):
        return [
            paddle.to_tensor([17], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_06f0f32abf283b86652fad44eb5a8550(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4674e5d6af96838b833215c3d8fb5cd1
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.362548828125, 0.337890625, 0.2119140625, 0.256103515625, 0.40234375, 0.251953125, 0.0684814453125, 0.059112548828125, 0.35400390625, 0.038787841796875, 0.4814453125, 0.322021484375, 0.1763916015625, 0.317138671875, 0.1461181640625, 0.1304931640625, 0.387939453125, 0.486328125, 0.45458984375, 0.1815185546875, 0.352783203125, 0.2122802734375, 0.24267578125, 0.2237548828125, 0.258544921875, 0.303955078125, 0.298583984375, 0.10137939453125, 0.228271484375, 0.007808685302734375]], dtype='float32').reshape([1, 30]),
            paddle.to_tensor([1.77113e+27], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f00d797c4f8a99d2452d94a5b5ba8de8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5515fd315ea737c77d6c99f7d91ae548
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.77113e+27], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6556ba305fe7f5cdc3bad60137656346(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4674e5d6af96838b833215c3d8fb5cd1
    def get_inputs(self):
        return [
            paddle.uniform([1, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.77113e+27], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0ed693a30ee217447bf3db6cdd218bb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5515fd315ea737c77d6c99f7d91ae548
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 501, 256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_292a273a3b47417f4af1e58248d8078b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fcd9a8bc34100fbec68cf8187c1064f
    def get_inputs(self):
        return [
            paddle.to_tensor([24], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_90af5270b37f0a9f8e2d55288951bd50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4674e5d6af96838b833215c3d8fb5cd1
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.14959219098091125, 0.44124141335487366, 0.19626769423484802, 0.21490056812763214, 0.1024792492389679, 0.11223723739385605, 0.19989579916000366, 0.3321531116962433, 0.00828117597848177, 0.18732500076293945, 0.173598051071167, 0.3275885283946991, 0.44143903255462646, 0.3192833662033081, 0.43958142399787903, 0.44314369559288025, 0.09090101718902588, 0.2770577669143677, 0.2141173630952835, 0.28024381399154663, 0.19103780388832092, 0.2808667719364166, 0.37479549646377563, 0.11526362597942352, 0.47682732343673706, 0.4560212194919586, 0.43395668268203735, 0.08317799866199493, 0.04807469621300697, 0.46327465772628784]], dtype='float32').reshape([1, 30]),
            paddle.to_tensor([1.77113e+27], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_37622af2af8295cea88222f672995d93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a37393250aba543f01d6f6284978207c
    def get_inputs(self):
        return [
            paddle.to_tensor([[2, 44]], dtype='int64').reshape([1, 2]),
            paddle.to_tensor([[2]], dtype='int64').reshape([1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3b27106b9bcf7d40ff223f62fc591dce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e1fe63f74b41f18eedcf35529ea2d6c
    def get_inputs(self):
        return [
            paddle.to_tensor([44], dtype='int64').reshape([1]),
            paddle.to_tensor([1.77113e+27], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a77f5b975b9037771b75b80fd62dfd6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d09cc67de54d383b7b6cae57e5372b12
    def get_inputs(self):
        return [
            paddle.to_tensor([0.17613212764263153], dtype='float32').reshape([1]),
            paddle.to_tensor([1.77113e+27], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_22d1f928d9860f6e049fcf14d760caad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4674e5d6af96838b833215c3d8fb5cd1
    def get_inputs(self):
        return [
            paddle.to_tensor([[1.0, 0.17613212764263153]], dtype='float32').reshape([1, 2]),
            paddle.to_tensor([[1.0]], dtype='float32').reshape([1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7d7f0cd6cd251c5e09a8658ae8e751c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf7ab9538118f29dbdd784fc9f71f720
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 30], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 501, 30], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0ac67a2b495a50b8526cc09fe1c49607(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fcd9a8bc34100fbec68cf8187c1064f
    def get_inputs(self):
        return [
            paddle.to_tensor([8], dtype='int32').reshape([1]),
            paddle.to_tensor([0], dtype='int32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c5b829de6eb85cfb2b98644b4b747c5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4674e5d6af96838b833215c3d8fb5cd1
    def get_inputs(self):
        return [
            paddle.to_tensor([[15.4140625, 15.4921875, 16.390625, 16.46875, 15.1875, 16.859375, 15.7578125, 15.15625, 17.0, 16.015625, 16.234375, 16.0, 15.8671875, 15.15625, 15.5390625, 16.734375, 16.296875, 16.53125, 16.25, 15.125, 15.2734375, 15.3984375, 16.796875, 15.7890625, 16.421875, 15.9375, 16.71875, 14.859375, 15.7890625, 16.203125]], dtype='float32').reshape([1, 30]),
            paddle.to_tensor([1.77113e+27], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_39bf7a21a8e51103d67ce94733e00cb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4674e5d6af96838b833215c3d8fb5cd1
    def get_inputs(self):
        return [
            paddle.to_tensor([[1.0, 1.0, 1.0, 1.0]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([1.77113e+27], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a4e17635836da6dc88cb2f9435a33d27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf7ab9538118f29dbdd784fc9f71f720
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 4], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 501, 4], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9634ed5bfa2dff41303115a6b91582df(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.assign_out_(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 501, 30], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 501, 30], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_17cebd63a27bd4a4ed7f403ba3638a04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9634ed5bfa2dff41303115a6b91582df
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 30], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 501, 30], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2fd44eb8496fa7326ef9947818869c0a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.assign_out_(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 30], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 30], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c65b284f92620f9315d52bba3ddeb645(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2fd44eb8496fa7326ef9947818869c0a
    def get_inputs(self):
        return [
            paddle.to_tensor([[16.59449005126953, 18.070903778076172, 16.7813720703125, 17.00531005859375, 16.58167266845703, 16.619869232177734, 15.407929420471191, 15.935876846313477, 16.032629013061523, 16.307723999023438, 16.820934295654297, 15.481019973754883, 16.534021377563477, 16.698375701904297, 17.18429183959961, 17.315828323364258, 15.136454582214355, 17.12547492980957, 17.039785385131836, 16.16747283935547, 16.435935974121094, 15.712438583374023, 17.211814880371094, 16.4127197265625, 17.575082778930664, 16.354190826416016, 17.133848190307617, 16.645544052124023, 16.288793563842773, 16.682811737060547]], dtype='float32').reshape([1, 30]),
            paddle.to_tensor([1.77113e+27], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2c96bbab6d742e0e3a2b29ec32513090(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.assign_out_(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_adc969e9f3e011058c6623781c681309(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c96bbab6d742e0e3a2b29ec32513090
    def get_inputs(self):
        return [
            paddle.uniform([1, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_53d730bfd03c8a5ee3fa96984cb3c4bc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.assign_out_(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2f81ee1c8d51e4fa05c816ce7ed3d4e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53d730bfd03c8a5ee3fa96984cb3c4bc
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.9999996423721313, 0.9999994039535522, 0.9999997615814209, 0.9999998807907104]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([1.77113e+27], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b5f3f934a273448aa68d2ace996f4197(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.assign_out_(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 501, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 501, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0753933a7f2a5f7f20b5cdbba182ad1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b5f3f934a273448aa68d2ace996f4197
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 4], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 501, 4], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5a284060b5cb90e7d1d074d777712102(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.assign_out_(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 40, 6, 40], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 40, 6, 40], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b239bd1c70ec890ea150a049e31bfd92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a284060b5cb90e7d1d074d777712102
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 40, 6, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a67e3ab7e7827322da1ca82b26f2a601(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.assign_out_(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f3822b5a59d616c9c29a42a45bc2e07d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a67e3ab7e7827322da1ca82b26f2a601
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.77113e+27], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_515c7a8787bb6db3f4e0ffa1130180e6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.assign_out_(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='int64'),
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9ec6cebc852a376594e9078ca73cfb9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_515c7a8787bb6db3f4e0ffa1130180e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[2, 0]], dtype='int64').reshape([1, 2]),
            paddle.to_tensor([[2]], dtype='int64').reshape([1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2fee5f7f55782e37b1e86813088390fd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.assign_out_(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_85c7d93e28110deed0e12b475fc883bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2fee5f7f55782e37b1e86813088390fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.77113e+27], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ca93a44a2d3899774c699ff7e4a9543a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.assign_out_(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 99], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 99], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b8a009cdf8dc2c3af2861af981f307bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca93a44a2d3899774c699ff7e4a9543a
    def get_inputs(self):
        return [
            paddle.uniform([1, 99], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.77113e+27], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0272ca864c66c6c5ebe005017a4f433c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.assign_out_(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 1, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cadd4d41cd97492b767c60a829ac40b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0272ca864c66c6c5ebe005017a4f433c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.0]]]], dtype='float32').reshape([1, 1, 1, 1]),
            paddle.to_tensor([1.77113e+27], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_044cc1e52b2e28fa3e5fe123ba4fa2fe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.assign_out_(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_afc3e1c8e88cfa83aa8545ae694b3cad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_044cc1e52b2e28fa3e5fe123ba4fa2fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1.0]], dtype='float32').reshape([1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_18331f9ea303e6f231453eeedac8cead(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.assign_out_(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 256], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_87699bf5eae1f472e45ade4818dd94e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_18331f9ea303e6f231453eeedac8cead
    def get_inputs(self):
        return [
            paddle.uniform([1, 256], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 256], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_55d99aec8c9eced3ab6925cb44a8a5a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2fd44eb8496fa7326ef9947818869c0a
    def get_inputs(self):
        return [
            paddle.to_tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype='float32').reshape([1, 30]),
            paddle.to_tensor([1.77113e+27], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d420ab22a0888b529d98bba41ace66e2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.assign_out_(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 501, 256], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 501, 256], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_22a9b8c416125b8bc5cdab7288b6363a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d420ab22a0888b529d98bba41ace66e2
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 256], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 501, 256], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6d3fe89cde5672fb2c7825a5e74ca638(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2fd44eb8496fa7326ef9947818869c0a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.362548828125, 0.337890625, 0.2119140625, 0.256103515625, 0.40234375, 0.251953125, 0.0684814453125, 0.059112548828125, 0.35400390625, 0.038787841796875, 0.4814453125, 0.322021484375, 0.1763916015625, 0.317138671875, 0.1461181640625, 0.1304931640625, 0.387939453125, 0.486328125, 0.45458984375, 0.1815185546875, 0.352783203125, 0.2122802734375, 0.24267578125, 0.2237548828125, 0.258544921875, 0.303955078125, 0.298583984375, 0.10137939453125, 0.228271484375, 0.007808685302734375]], dtype='float32').reshape([1, 30]),
            paddle.to_tensor([1.77113e+27], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b21096f50034af80a1ff7525be5ff762(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.assign_out_(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_aee7e1186aad2ef3a6288f2ea8ff3981(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b21096f50034af80a1ff7525be5ff762
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.77113e+27], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8a9900803e833d70b21e3a9606568366(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c96bbab6d742e0e3a2b29ec32513090
    def get_inputs(self):
        return [
            paddle.uniform([1, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1.77113e+27], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0e3664f6c310c1502934886090b2ff2f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.assign_out_(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 501, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 501, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_14ea9c432899d8c83bf5636fbd03ecaf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e3664f6c310c1502934886090b2ff2f
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 501, 256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_54859aeae93f1de97a12c13f099f9b42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2fd44eb8496fa7326ef9947818869c0a
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.14959219098091125, 0.44124141335487366, 0.19626769423484802, 0.21490056812763214, 0.1024792492389679, 0.11223723739385605, 0.19989579916000366, 0.3321531116962433, 0.00828117597848177, 0.18732500076293945, 0.173598051071167, 0.3275885283946991, 0.44143903255462646, 0.3192833662033081, 0.43958142399787903, 0.44314369559288025, 0.09090101718902588, 0.2770577669143677, 0.2141173630952835, 0.28024381399154663, 0.19103780388832092, 0.2808667719364166, 0.37479549646377563, 0.11526362597942352, 0.47682732343673706, 0.4560212194919586, 0.43395668268203735, 0.08317799866199493, 0.04807469621300697, 0.46327465772628784]], dtype='float32').reshape([1, 30]),
            paddle.to_tensor([1.77113e+27], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_648035e5212db0c7229c6cda6f0f7893(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_515c7a8787bb6db3f4e0ffa1130180e6
    def get_inputs(self):
        return [
            paddle.to_tensor([[2, 44]], dtype='int64').reshape([1, 2]),
            paddle.to_tensor([[2]], dtype='int64').reshape([1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_efe939539805cb652fbe780b20ac68b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_044cc1e52b2e28fa3e5fe123ba4fa2fe
    def get_inputs(self):
        return [
            paddle.to_tensor([[1.0, 0.17613212764263153]], dtype='float32').reshape([1, 2]),
            paddle.to_tensor([[1.0]], dtype='float32').reshape([1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ffde6a438a32304438f1aa68a95dbead(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.assign_out_(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 501, 30], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 501, 30], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b7b5f53bf66ee7a438a88d5ef5603c7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ffde6a438a32304438f1aa68a95dbead
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 30], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 501, 30], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6a555c216cb0a1d46c82f5211567a4f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2fd44eb8496fa7326ef9947818869c0a
    def get_inputs(self):
        return [
            paddle.to_tensor([[15.4140625, 15.4921875, 16.390625, 16.46875, 15.1875, 16.859375, 15.7578125, 15.15625, 17.0, 16.015625, 16.234375, 16.0, 15.8671875, 15.15625, 15.5390625, 16.734375, 16.296875, 16.53125, 16.25, 15.125, 15.2734375, 15.3984375, 16.796875, 15.7890625, 16.421875, 15.9375, 16.71875, 14.859375, 15.7890625, 16.203125]], dtype='float32').reshape([1, 30]),
            paddle.to_tensor([1.77113e+27], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_41bff02171c652b18b55ee399e5139c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53d730bfd03c8a5ee3fa96984cb3c4bc
    def get_inputs(self):
        return [
            paddle.to_tensor([[1.0, 1.0, 1.0, 1.0]], dtype='float32').reshape([1, 4]),
            paddle.to_tensor([1.77113e+27], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6ea90cb997f4b5bc1012640b0aaf77a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.assign_out_(input_0, input_1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 501, 4], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 501, 4], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6fccdf75a5cf4ad0dc979acc3047f894(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ea90cb997f4b5bc1012640b0aaf77a6
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 4], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 501, 4], dtype='float16', min=0, max=0.5),
        ]


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