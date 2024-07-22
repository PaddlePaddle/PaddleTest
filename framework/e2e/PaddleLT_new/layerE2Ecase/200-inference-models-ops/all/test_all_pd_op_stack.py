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
class PrimitiveOp_5d58f36745a9dfac3d783601131351ea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        arg_0_0 = paddle._C_ops.full_int_array(1, paddle.int32, paddle.core.CPUPlace())
        arg_0_1 = paddle._C_ops.full_int_array(256, paddle.int32, paddle.core.CPUPlace())
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_772efabf2924918fd83f2656d1ca5bcc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d58f36745a9dfac3d783601131351ea
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor(256, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d85f49f447feb0a3bb6954cf9a9e925c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        arg_0_0 = paddle._C_ops.full_int_array(1, paddle.int32, paddle.core.CPUPlace())
        arg_0_1 = paddle._C_ops.full_int_array(501, paddle.int32, paddle.core.CPUPlace())
        arg_0_2 = paddle._C_ops.full_int_array(256, paddle.int32, paddle.core.CPUPlace())
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        return paddle._C_ops.stack(input_0, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e82c1b23ff4f009f7318781a6cf870ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d85f49f447feb0a3bb6954cf9a9e925c
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor(501, dtype='int32').reshape([]),
            paddle.to_tensor(256, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8f61c79ad9c54f8ce5d75f8bb3a48bd1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0):
        arg_0_0 = paddle._C_ops.full_int_array(1, paddle.int32, paddle.core.CPUPlace())
        input_0 = [arg_0_0]
        return paddle._C_ops.stack(input_0, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_474ee661942123b3b828cb1293a4ca82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f61c79ad9c54f8ce5d75f8bb3a48bd1
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6f145b26eb2686322e68b40356dd8734(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        arg_0_0 = paddle._C_ops.full_int_array(1, paddle.int32, paddle.core.CPUPlace())
        arg_0_1 = paddle._C_ops.full_int_array(501, paddle.int32, paddle.core.CPUPlace())
        arg_0_2 = paddle._C_ops.full_int_array(30, paddle.int32, paddle.core.CPUPlace())
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        return paddle._C_ops.stack(input_0, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a949a11bfa7585e201c65da3e4d191e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f145b26eb2686322e68b40356dd8734
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor(501, dtype='int32').reshape([]),
            paddle.to_tensor(30, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1dc077ea84ad4b512985bd109693cbeb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        arg_0_0 = paddle._C_ops.full_int_array(1, paddle.int32, paddle.core.CPUPlace())
        arg_0_1 = paddle._C_ops.full_int_array(501, paddle.int32, paddle.core.CPUPlace())
        arg_0_2 = paddle._C_ops.full_int_array(4, paddle.int32, paddle.core.CPUPlace())
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        return paddle._C_ops.stack(input_0, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_56d485eda54d52acd1aa6593203107ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1dc077ea84ad4b512985bd109693cbeb
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor(501, dtype='int32').reshape([]),
            paddle.to_tensor(4, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_650dedbe87862afc2fcfc017666cf1ba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float64'),
            paddle.static.InputSpec(shape=[None], dtype='float64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_71a5756caafc0f45f1d6f4d57d2fc8e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_650dedbe87862afc2fcfc017666cf1ba
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.7777777777777778, -0.5555555555555556, -0.33333333333333337, -0.11111111111111116, 0.11111111111111116, 0.33333333333333337, 0.5555555555555556, 0.7777777777777778, 1.0], dtype='float64').reshape([10]),
            paddle.to_tensor([0.01941630376076567, 0.3790526156068944, 0.3593904017742824, 0.002979560754593019, 0.2384611628863936, 0.2182505042141605, 0.22999032513806358, 0.44186109374094196, 0.30590534300596195, 0.0854299715880357], dtype='float64').reshape([10]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bca081edeadea9e225d06675a6de36f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_650dedbe87862afc2fcfc017666cf1ba
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.7777777777777778, -0.5555555555555556, -0.33333333333333337, -0.11111111111111116, 0.11111111111111116, 0.33333333333333337, 0.5555555555555556, 0.7777777777777778, 1.0], dtype='float64').reshape([10]),
            paddle.to_tensor([0.47432414810209067, 0.1136666916729526, 0.29019240376952005, 0.020012576729375703, 0.3240750423366652, 0.36988926380293796, 0.07155616009722401, 0.19599883502397977, 0.2189904546934293, 0.3785597512223029], dtype='float64').reshape([10]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c5a9cd481ddfb12d017f1691531161d9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float64'),
            paddle.static.InputSpec(shape=[None, None], dtype='float64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_83ba48cc06c504b721d882f736e25245(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5a9cd481ddfb12d017f1691531161d9
    def get_inputs(self):
        return [
            paddle.uniform([100, 32], dtype='float64', min=0, max=0.5),
            paddle.uniform([100, 32], dtype='float64', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_55aebdfbdcc23fe1f04e9d3c48716eef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        arg_0_0 = paddle._C_ops.full_int_array(4, paddle.int32, paddle.core.CPUPlace())
        arg_0_1 = paddle._C_ops.full_int_array(1, paddle.int32, paddle.core.CPUPlace())
        arg_0_2 = paddle._C_ops.full_int_array(96, paddle.int32, paddle.core.CPUPlace())
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        return paddle._C_ops.stack(input_0, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0d7a8daf3d6a08bf9844c60cb7e221b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_55aebdfbdcc23fe1f04e9d3c48716eef
    def get_inputs(self):
        return [
            paddle.to_tensor(4, dtype='int32').reshape([]),
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor(96, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a6f42a04d95ba777d8dad4b6767ede34(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        arg_0_0 = paddle._C_ops.full_int_array(1, paddle.int32, paddle.core.CPUPlace())
        arg_0_1 = paddle._C_ops.full_int_array(96, paddle.int32, paddle.core.CPUPlace())
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_81f53d2bf1f87b69b19a06173fc08324(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6f42a04d95ba777d8dad4b6767ede34
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor(96, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_26acc0f0b9eded328198dc5adfe615bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_650dedbe87862afc2fcfc017666cf1ba
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.7777777777777778, -0.5555555555555556, -0.33333333333333337, -0.11111111111111116, 0.11111111111111116, 0.33333333333333337, 0.5555555555555556, 0.7777777777777778, 1.0], dtype='float64').reshape([10]),
            paddle.to_tensor([0.3077561729212035, 0.3457365679870965, 0.046071192079243956, 0.49958929433142074, 0.2333132556728729, 0.2577827069251193, 0.10530989502197233, 0.15038999064318617, 0.08911398431114073, 0.31019259296028256], dtype='float64').reshape([10]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8258303d84962b924392e4cc46733963(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_650dedbe87862afc2fcfc017666cf1ba
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.7777777777777778, -0.5555555555555556, -0.33333333333333337, -0.11111111111111116, 0.11111111111111116, 0.33333333333333337, 0.5555555555555556, 0.7777777777777778, 1.0], dtype='float64').reshape([10]),
            paddle.to_tensor([0.3940580851867268, 0.0943041360743195, 0.2239685264623252, 0.2975335214163628, 0.05894664109751255, 0.13609892524784048, 0.27656972252663803, 0.24543013726329818, 0.3446411272898572, 0.09662762334195354], dtype='float64').reshape([10]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_55dff704f514640cc01a70a08e20ef2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_650dedbe87862afc2fcfc017666cf1ba
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.7777777777777778, -0.5555555555555556, -0.33333333333333337, -0.11111111111111116, 0.11111111111111116, 0.33333333333333337, 0.5555555555555556, 0.7777777777777778, 1.0], dtype='float64').reshape([10]),
            paddle.to_tensor([0.03707604432307107, 0.3700790015440113, 0.2749804754232069, 0.2810508170681666, 0.08436610491149285, 0.3538608780693444, 0.19009636664160842, 0.1662973248165914, 0.4495971912959372, 0.2968036209835915], dtype='float64').reshape([10]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f2b2ce8067379052f81f1d6a1f49cb3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_650dedbe87862afc2fcfc017666cf1ba
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.7777777777777778, -0.5555555555555556, -0.33333333333333337, -0.11111111111111116, 0.11111111111111116, 0.33333333333333337, 0.5555555555555556, 0.7777777777777778, 1.0], dtype='float64').reshape([10]),
            paddle.to_tensor([0.3167038075148165, 0.2982564067123059, 0.45331366090556324, 0.02654642015799394, 0.17896874650378605, 0.16640199727742358, 0.30339663886546986, 0.06845033480486609, 0.13162286738958287, 0.1562495873457828], dtype='float64').reshape([10]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f25f31a1228793e1f295c1ed1dc08448(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        arg_0_0 = paddle._C_ops.full_int_array(1, paddle.int32, paddle.core.CPUPlace())
        arg_0_1 = paddle._C_ops.full_int_array(1, paddle.int32, paddle.core.CPUPlace())
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_69ebe8460a78e9b740af86b282b12f60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f25f31a1228793e1f295c1ed1dc08448
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor(1, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_85df9c88294fa8d3bfe9f3bf2ce88b5d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        arg_0_0 = paddle._C_ops.full_int_array(1, paddle.int32, paddle.core.CPUPlace())
        arg_0_1 = paddle._C_ops.full_int_array(26, paddle.int32, paddle.core.CPUPlace())
        arg_0_2 = paddle._C_ops.full_int_array(512, paddle.int32, paddle.core.CPUPlace())
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        return paddle._C_ops.stack(input_0, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ce8071d1fcb1bb1701fd1d2a850ce662(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85df9c88294fa8d3bfe9f3bf2ce88b5d
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor(26, dtype='int32').reshape([]),
            paddle.to_tensor(512, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b198380ff5bf81dd4eb3196cfae27820(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        arg_0_0 = paddle._C_ops.full_int_array(1, paddle.int32, paddle.core.CPUPlace())
        arg_0_1 = paddle._C_ops.full_int_array(26, paddle.int32, paddle.core.CPUPlace())
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_75b3059a6c879690768c814722035293(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b198380ff5bf81dd4eb3196cfae27820
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor(26, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1107d03777e37b507c55bb1f1032f054(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_650dedbe87862afc2fcfc017666cf1ba
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.7777777777777778, -0.5555555555555556, -0.33333333333333337, -0.11111111111111116, 0.11111111111111116, 0.33333333333333337, 0.5555555555555556, 0.7777777777777778, 1.0], dtype='float64').reshape([10]),
            paddle.to_tensor([0.34446578771082015, 0.2562524998809719, 0.4481117816643626, 0.1619197367177064, 0.17501214789944905, 0.3418746131024314, 0.3085521583075793, 0.32479860823096207, 0.3237859331170114, 0.2578180375120206], dtype='float64').reshape([10]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_938ddcd290970e71049ff5a9fc96baae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_650dedbe87862afc2fcfc017666cf1ba
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.7777777777777778, -0.5555555555555556, -0.33333333333333337, -0.11111111111111116, 0.11111111111111116, 0.33333333333333337, 0.5555555555555556, 0.7777777777777778, 1.0], dtype='float64').reshape([10]),
            paddle.to_tensor([0.07801709318059605, 0.26994842332776336, 0.4573174968209762, 0.25705318620870377, 0.4111060282467436, 0.08470458214390955, 0.1231947742008748, 0.3430186835407585, 0.15580186771618995, 0.1322347264714909], dtype='float64').reshape([10]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_cf44d9576debdbb6ba098276bc9990c8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        arg_0_0 = paddle._C_ops.full_int_array(1, paddle.int32, paddle.core.CPUPlace())
        arg_0_1 = paddle._C_ops.full_int_array(40, paddle.int32, paddle.core.CPUPlace())
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e073f1fa1332806c2b42c4d7f3449753(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf44d9576debdbb6ba098276bc9990c8
    def get_inputs(self):
        return [
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor(40, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_802152bc292acb3e6545b585f6ad535a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_650dedbe87862afc2fcfc017666cf1ba
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.7777777777777778, -0.5555555555555556, -0.33333333333333337, -0.11111111111111116, 0.11111111111111116, 0.33333333333333337, 0.5555555555555556, 0.7777777777777778, 1.0], dtype='float64').reshape([10]),
            paddle.to_tensor([0.31603998750400486, 0.2785113351083266, 0.3385784360357479, 0.030717021930404015, 0.12799757500969675, 0.15166574045688838, 0.4490317614615763, 0.060374541068107995, 0.368821132291721, 0.0033895628611893824], dtype='float64').reshape([10]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2afbb2afdea69017f04e40292b8410e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_650dedbe87862afc2fcfc017666cf1ba
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.7777777777777778, -0.5555555555555556, -0.33333333333333337, -0.11111111111111116, 0.11111111111111116, 0.33333333333333337, 0.5555555555555556, 0.7777777777777778, 1.0], dtype='float64').reshape([10]),
            paddle.to_tensor([0.2050675678120614, 0.3821714508965983, 0.1695591506675583, 0.43937429080247015, 0.07648084588565321, 0.49312595809878623, 0.3813546802907153, 0.42707595477801497, 0.23965132927011526, 0.06716404423990432], dtype='float64').reshape([10]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ab157d187a43382ed9becd90d46416bc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12]
        return paddle._C_ops.stack(input_0, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6e43a1f9db86969169373660e3bdde5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab157d187a43382ed9becd90d46416bc
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e30bcfc9e371eccfed479bf0583a56e2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2):
        arg_0_0 = paddle._C_ops.full_int_array(2, paddle.int32, paddle.core.CPUPlace())
        arg_0_1 = paddle._C_ops.full_int_array(1, paddle.int32, paddle.core.CPUPlace())
        arg_0_2 = paddle._C_ops.full_int_array(256, paddle.int32, paddle.core.CPUPlace())
        input_0 = [arg_0_0, arg_0_1, arg_0_2]
        return paddle._C_ops.stack(input_0, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_89c5d68c5880622e2db2360b1975a0f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e30bcfc9e371eccfed479bf0583a56e2
    def get_inputs(self):
        return [
            paddle.to_tensor(2, dtype='int32').reshape([]),
            paddle.to_tensor(1, dtype='int32').reshape([]),
            paddle.to_tensor(256, dtype='int32').reshape([]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3ed18eda275e4f68554fe978315bebe8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_650dedbe87862afc2fcfc017666cf1ba
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.7777777777777778, -0.5555555555555556, -0.33333333333333337, -0.11111111111111116, 0.11111111111111116, 0.33333333333333337, 0.5555555555555556, 0.7777777777777778, 1.0], dtype='float64').reshape([10]),
            paddle.to_tensor([0.25778206813796134, 0.47212243781132057, 0.17757820341255418, 0.1577704766150869, 0.2729343939848382, 0.23934846371573557, 0.33981630809894214, 0.25548244368504447, 0.32611274904295484, 0.4534974883884977], dtype='float64').reshape([10]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9bfea70196040bccbd78dbadda5cfea7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_650dedbe87862afc2fcfc017666cf1ba
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.7777777777777778, -0.5555555555555556, -0.33333333333333337, -0.11111111111111116, 0.11111111111111116, 0.33333333333333337, 0.5555555555555556, 0.7777777777777778, 1.0], dtype='float64').reshape([10]),
            paddle.to_tensor([0.13316575888118973, 0.3336867725120918, 0.21704645306568707, 0.2886064676065506, 0.11163694091000978, 0.13298269634465623, 0.22492829847098886, 0.032865739001113475, 0.10582055291656448, 0.14867209601655676], dtype='float64').reshape([10]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_450e50b5b2c41128ddaafc1388186e9b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12]
        return paddle._C_ops.stack(input_0, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e371f8a991f8dff12d50f5965d7bc82f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_450e50b5b2c41128ddaafc1388186e9b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9629cc98e1893616e9e54ec96c0f1ed7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8a8cb107b8ef131557229ad4cb4fc08c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9629cc98e1893616e9e54ec96c0f1ed7
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[48, 80], dtype='int64'), 'int64'),
            paddle.cast(paddle.randint(low=0, high=3, shape=[48, 80], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b82a934ec9f0a629bd64336b032cefef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 3)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_555fa9e40c57688b5d69a8dc10d30334(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b82a934ec9f0a629bd64336b032cefef
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ca2a7339fdfea6cfacd9ca489a04fa0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9629cc98e1893616e9e54ec96c0f1ed7
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[96, 160], dtype='int64'), 'int64'),
            paddle.cast(paddle.randint(low=0, high=3, shape=[96, 160], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7bc4f8e69482d7917295aff609444258(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b82a934ec9f0a629bd64336b032cefef
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_826d8c46ac0b5d55189f68f45eae0989(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9629cc98e1893616e9e54ec96c0f1ed7
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[192, 320], dtype='int64'), 'int64'),
            paddle.cast(paddle.randint(low=0, high=3, shape=[192, 320], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6fb9f9ffa50529db22615cf223e72100(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b82a934ec9f0a629bd64336b032cefef
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0db994d38a08ad1db4419e5f9c42af90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9629cc98e1893616e9e54ec96c0f1ed7
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[180, 320], dtype='int64'), 'int64'),
            paddle.cast(paddle.randint(low=0, high=3, shape=[180, 320], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5592740ace4c7134d789dc9b50f242e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b82a934ec9f0a629bd64336b032cefef
    def get_inputs(self):
        return [
            paddle.uniform([1, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 180, 320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_3ca1131b5cd2d430cd169e5e4f5f5461(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4e39cdba247c98bd467105e7a37791de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ca1131b5cd2d430cd169e5e4f5f5461
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 720, 1280], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 720, 1280], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_71cc0f07db57866d77e7b57c3e2babf9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_20323862a65777b9d20fe541bf7a059c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71cc0f07db57866d77e7b57c3e2babf9
    def get_inputs(self):
        return [
            paddle.uniform([80, 80], dtype='float16', min=0, max=0.5),
            paddle.uniform([80, 80], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1fc004a8ba7dad2a9493d09a8c64d11f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71cc0f07db57866d77e7b57c3e2babf9
    def get_inputs(self):
        return [
            paddle.uniform([40, 40], dtype='float16', min=0, max=0.5),
            paddle.uniform([40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7633545e5e89551273d16ae32ffd6d6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71cc0f07db57866d77e7b57c3e2babf9
    def get_inputs(self):
        return [
            paddle.uniform([20, 20], dtype='float16', min=0, max=0.5),
            paddle.uniform([20, 20], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_7ff54aeb46e416d7a815006a705ca84a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 3)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fa88a72684524776e9c5fdc7dec5b315(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ff54aeb46e416d7a815006a705ca84a
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 80], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 48, 80], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_663f5b0c5c936b8b341bd6e2ee43cf10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ff54aeb46e416d7a815006a705ca84a
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 160], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 96, 160], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f837dfae74092b61ebf9c740c09113b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ff54aeb46e416d7a815006a705ca84a
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 320], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 192, 320], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1b4052b247b9657499e255a90d7ad3fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ff54aeb46e416d7a815006a705ca84a
    def get_inputs(self):
        return [
            paddle.uniform([1, 180, 320], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 180, 320], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_309d6894ea5bfd1f15619c57aeda98ae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d57376d4cbb1b5a4d40adbae02bcbda9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_309d6894ea5bfd1f15619c57aeda98ae
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 720, 1280], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 3, 720, 1280], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f4562a3fdc807d6e1bd4d5065f470726(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ad30119ebf8da2dd9ae504a140b89f3f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4562a3fdc807d6e1bd4d5065f470726
    def get_inputs(self):
        return [
            paddle.uniform([80, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9391f12ab2406f96940a794c47af42ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4562a3fdc807d6e1bd4d5065f470726
    def get_inputs(self):
        return [
            paddle.uniform([40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a4dfed59c9c6e564bc00340c4d0357bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4562a3fdc807d6e1bd4d5065f470726
    def get_inputs(self):
        return [
            paddle.uniform([20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([20, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a9f0881d443ed9edbdd48a49615a839a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[10], dtype='float64'),
            paddle.static.InputSpec(shape=[10], dtype='float64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5d099a30a9c16c244a9ac7d5ebe2c607(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9f0881d443ed9edbdd48a49615a839a
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.7777777777777778, -0.5555555555555556, -0.33333333333333337, -0.11111111111111116, 0.11111111111111116, 0.33333333333333337, 0.5555555555555556, 0.7777777777777778, 1.0], dtype='float64').reshape([10]),
            paddle.to_tensor([0.01941630376076567, 0.3790526156068944, 0.3593904017742824, 0.002979560754593019, 0.2384611628863936, 0.2182505042141605, 0.22999032513806358, 0.44186109374094196, 0.30590534300596195, 0.0854299715880357], dtype='float64').reshape([10]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1f4b1985e9763c4127c35118617865f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9f0881d443ed9edbdd48a49615a839a
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.7777777777777778, -0.5555555555555556, -0.33333333333333337, -0.11111111111111116, 0.11111111111111116, 0.33333333333333337, 0.5555555555555556, 0.7777777777777778, 1.0], dtype='float64').reshape([10]),
            paddle.to_tensor([0.47432414810209067, 0.1136666916729526, 0.29019240376952005, 0.020012576729375703, 0.3240750423366652, 0.36988926380293796, 0.07155616009722401, 0.19599883502397977, 0.2189904546934293, 0.3785597512223029], dtype='float64').reshape([10]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1fd84281c9c11cc6c918c79c3f7071a3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, 32], dtype='float64'),
            paddle.static.InputSpec(shape=[100, 32], dtype='float64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_95797d421e5de1bf11a050038ce38592(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fd84281c9c11cc6c918c79c3f7071a3
    def get_inputs(self):
        return [
            paddle.uniform([100, 32], dtype='float64', min=0, max=0.5),
            paddle.uniform([100, 32], dtype='float64', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_31531f7d6633bed9b99897c4b2155efd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9f0881d443ed9edbdd48a49615a839a
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.7777777777777778, -0.5555555555555556, -0.33333333333333337, -0.11111111111111116, 0.11111111111111116, 0.33333333333333337, 0.5555555555555556, 0.7777777777777778, 1.0], dtype='float64').reshape([10]),
            paddle.to_tensor([0.3077561729212035, 0.3457365679870965, 0.046071192079243956, 0.49958929433142074, 0.2333132556728729, 0.2577827069251193, 0.10530989502197233, 0.15038999064318617, 0.08911398431114073, 0.31019259296028256], dtype='float64').reshape([10]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_46d105903772097d2f1a2220c3960f22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9f0881d443ed9edbdd48a49615a839a
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.7777777777777778, -0.5555555555555556, -0.33333333333333337, -0.11111111111111116, 0.11111111111111116, 0.33333333333333337, 0.5555555555555556, 0.7777777777777778, 1.0], dtype='float64').reshape([10]),
            paddle.to_tensor([0.3940580851867268, 0.0943041360743195, 0.2239685264623252, 0.2975335214163628, 0.05894664109751255, 0.13609892524784048, 0.27656972252663803, 0.24543013726329818, 0.3446411272898572, 0.09662762334195354], dtype='float64').reshape([10]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_952cecfcadebd5d1c4835263584c2508(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9f0881d443ed9edbdd48a49615a839a
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.7777777777777778, -0.5555555555555556, -0.33333333333333337, -0.11111111111111116, 0.11111111111111116, 0.33333333333333337, 0.5555555555555556, 0.7777777777777778, 1.0], dtype='float64').reshape([10]),
            paddle.to_tensor([0.03707604432307107, 0.3700790015440113, 0.2749804754232069, 0.2810508170681666, 0.08436610491149285, 0.3538608780693444, 0.19009636664160842, 0.1662973248165914, 0.4495971912959372, 0.2968036209835915], dtype='float64').reshape([10]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e8c74d40a186fbbcae2b63615f5f05bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9f0881d443ed9edbdd48a49615a839a
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.7777777777777778, -0.5555555555555556, -0.33333333333333337, -0.11111111111111116, 0.11111111111111116, 0.33333333333333337, 0.5555555555555556, 0.7777777777777778, 1.0], dtype='float64').reshape([10]),
            paddle.to_tensor([0.3167038075148165, 0.2982564067123059, 0.45331366090556324, 0.02654642015799394, 0.17896874650378605, 0.16640199727742358, 0.30339663886546986, 0.06845033480486609, 0.13162286738958287, 0.1562495873457828], dtype='float64').reshape([10]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ddddfbacad4c1a54f8d8f9c980963f5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9f0881d443ed9edbdd48a49615a839a
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.7777777777777778, -0.5555555555555556, -0.33333333333333337, -0.11111111111111116, 0.11111111111111116, 0.33333333333333337, 0.5555555555555556, 0.7777777777777778, 1.0], dtype='float64').reshape([10]),
            paddle.to_tensor([0.34446578771082015, 0.2562524998809719, 0.4481117816643626, 0.1619197367177064, 0.17501214789944905, 0.3418746131024314, 0.3085521583075793, 0.32479860823096207, 0.3237859331170114, 0.2578180375120206], dtype='float64').reshape([10]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5a09fbb0be22af288e397198456052ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9f0881d443ed9edbdd48a49615a839a
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.7777777777777778, -0.5555555555555556, -0.33333333333333337, -0.11111111111111116, 0.11111111111111116, 0.33333333333333337, 0.5555555555555556, 0.7777777777777778, 1.0], dtype='float64').reshape([10]),
            paddle.to_tensor([0.07801709318059605, 0.26994842332776336, 0.4573174968209762, 0.25705318620870377, 0.4111060282467436, 0.08470458214390955, 0.1231947742008748, 0.3430186835407585, 0.15580186771618995, 0.1322347264714909], dtype='float64').reshape([10]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ce908e316ef1d1956a3fa8ff5210af1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9f0881d443ed9edbdd48a49615a839a
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.7777777777777778, -0.5555555555555556, -0.33333333333333337, -0.11111111111111116, 0.11111111111111116, 0.33333333333333337, 0.5555555555555556, 0.7777777777777778, 1.0], dtype='float64').reshape([10]),
            paddle.to_tensor([0.31603998750400486, 0.2785113351083266, 0.3385784360357479, 0.030717021930404015, 0.12799757500969675, 0.15166574045688838, 0.4490317614615763, 0.060374541068107995, 0.368821132291721, 0.0033895628611893824], dtype='float64').reshape([10]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_92108380120d2803f95b9300c6b17a59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9f0881d443ed9edbdd48a49615a839a
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.7777777777777778, -0.5555555555555556, -0.33333333333333337, -0.11111111111111116, 0.11111111111111116, 0.33333333333333337, 0.5555555555555556, 0.7777777777777778, 1.0], dtype='float64').reshape([10]),
            paddle.to_tensor([0.2050675678120614, 0.3821714508965983, 0.1695591506675583, 0.43937429080247015, 0.07648084588565321, 0.49312595809878623, 0.3813546802907153, 0.42707595477801497, 0.23965132927011526, 0.06716404423990432], dtype='float64').reshape([10]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_523c0ce1a82b41a8460fcf2edfe67bf7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12]
        return paddle._C_ops.stack(input_0, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5b0d3526af59ce26a825dbfb04e81bba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_523c0ce1a82b41a8460fcf2edfe67bf7
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_67bc2d80cd96e89ef6d613e66c1a3209(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9f0881d443ed9edbdd48a49615a839a
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.7777777777777778, -0.5555555555555556, -0.33333333333333337, -0.11111111111111116, 0.11111111111111116, 0.33333333333333337, 0.5555555555555556, 0.7777777777777778, 1.0], dtype='float64').reshape([10]),
            paddle.to_tensor([0.25778206813796134, 0.47212243781132057, 0.17757820341255418, 0.1577704766150869, 0.2729343939848382, 0.23934846371573557, 0.33981630809894214, 0.25548244368504447, 0.32611274904295484, 0.4534974883884977], dtype='float64').reshape([10]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ddf03fcd83cae3f0db1f50e97fa9076a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9f0881d443ed9edbdd48a49615a839a
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.7777777777777778, -0.5555555555555556, -0.33333333333333337, -0.11111111111111116, 0.11111111111111116, 0.33333333333333337, 0.5555555555555556, 0.7777777777777778, 1.0], dtype='float64').reshape([10]),
            paddle.to_tensor([0.13316575888118973, 0.3336867725120918, 0.21704645306568707, 0.2886064676065506, 0.11163694091000978, 0.13298269634465623, 0.22492829847098886, 0.032865739001113475, 0.10582055291656448, 0.14867209601655676], dtype='float64').reshape([10]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_147c479eab55ebcfbf3a3e4de5cfc8b7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12):
        input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7, arg_0_8, arg_0_9, arg_0_10, arg_0_11, arg_0_12]
        return paddle._C_ops.stack(input_0, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8356a4f6e2979aaf02432724d3596eab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_147c479eab55ebcfbf3a3e4de5cfc8b7
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_418bebd5ed0d22d6398928dc48fa3c24(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[48, 80], dtype='int64'),
            paddle.static.InputSpec(shape=[48, 80], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9a9a137b656da29425b64ea1f053f604(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_418bebd5ed0d22d6398928dc48fa3c24
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[48, 80], dtype='int64'), 'int64'),
            paddle.cast(paddle.randint(low=0, high=3, shape=[48, 80], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_3b6a7b9b261e5b541a8b29ede926062f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 3)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 48, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 48, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_14800072a33097727740fc71e566e80e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b6a7b9b261e5b541a8b29ede926062f
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 48, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4392290d59c69f23ebb30b027a3d0c52(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[96, 160], dtype='int64'),
            paddle.static.InputSpec(shape=[96, 160], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7dd8c381b40caa43e5cf5159ae09a38e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4392290d59c69f23ebb30b027a3d0c52
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[96, 160], dtype='int64'), 'int64'),
            paddle.cast(paddle.randint(low=0, high=3, shape=[96, 160], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f32ae5045a41c4238d56c54574cd6f0c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 3)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 160], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 96, 160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c1cc082b081a963eba0b7914c4139b3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f32ae5045a41c4238d56c54574cd6f0c
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 96, 160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_7f7f836c434bc20a1c5e47b516b0db51(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[192, 320], dtype='int64'),
            paddle.static.InputSpec(shape=[192, 320], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bf82a145cb65001e5146503b921b8a45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f7f836c434bc20a1c5e47b516b0db51
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[192, 320], dtype='int64'), 'int64'),
            paddle.cast(paddle.randint(low=0, high=3, shape=[192, 320], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f9fe0e1ceaef41d2fcf42c6326d90d05(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 3)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 192, 320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f248088f958bafb1ba869d05679d0d60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9fe0e1ceaef41d2fcf42c6326d90d05
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 192, 320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_506dd900d05f31945c1aeedf8e8a43e1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[180, 320], dtype='int64'),
            paddle.static.InputSpec(shape=[180, 320], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2f747ba44f41222f85793b66cd61489f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_506dd900d05f31945c1aeedf8e8a43e1
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[180, 320], dtype='int64'), 'int64'),
            paddle.cast(paddle.randint(low=0, high=3, shape=[180, 320], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e7af695db20fb55c8e46ddc1aa8f29dd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 3)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 180, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 180, 320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_72d594b6bdee2618a140e48ad3e054f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e7af695db20fb55c8e46ddc1aa8f29dd
    def get_inputs(self):
        return [
            paddle.uniform([1, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 180, 320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8d3d5d62ad425335232bd62a46c47c4f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 720, 1280], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 3, 720, 1280], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fdef9ffc87f50c90add3e04d3a9c559d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d3d5d62ad425335232bd62a46c47c4f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 720, 1280], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 3, 720, 1280], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2af03150064b421b2d71c9274c1334f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[80, 80], dtype='float16'),
            paddle.static.InputSpec(shape=[80, 80], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9f94e6565a3fa5d7dcb1b81fc8380b8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2af03150064b421b2d71c9274c1334f9
    def get_inputs(self):
        return [
            paddle.uniform([80, 80], dtype='float16', min=0, max=0.5),
            paddle.uniform([80, 80], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_14b9d8333d0e3c3bf2246fc9026a31dd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[40, 40], dtype='float16'),
            paddle.static.InputSpec(shape=[40, 40], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_601e52982ede487ee13f87a619863f40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_14b9d8333d0e3c3bf2246fc9026a31dd
    def get_inputs(self):
        return [
            paddle.uniform([40, 40], dtype='float16', min=0, max=0.5),
            paddle.uniform([40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a53aea7634d7c02d587aef449b840654(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20, 20], dtype='float16'),
            paddle.static.InputSpec(shape=[20, 20], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3d94e496080dc3e57a5aeffae8a4f594(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a53aea7634d7c02d587aef449b840654
    def get_inputs(self):
        return [
            paddle.uniform([20, 20], dtype='float16', min=0, max=0.5),
            paddle.uniform([20, 20], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_57658105089dfcedff4bca6b1ebd7add(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 3)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 48, 80], dtype='float16'),
            paddle.static.InputSpec(shape=[1, 48, 80], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0b6cd05a09a498f7129a181ea28fb3fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57658105089dfcedff4bca6b1ebd7add
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 80], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 48, 80], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_054414d8920c6ae89837152ddcd3c060(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 3)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 160], dtype='float16'),
            paddle.static.InputSpec(shape=[1, 96, 160], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fa3388218816b8bdd8d63c228860fa8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_054414d8920c6ae89837152ddcd3c060
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 160], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 96, 160], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_08d1fc531e9cffbde81f0f4b433cb9c8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 3)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 320], dtype='float16'),
            paddle.static.InputSpec(shape=[1, 192, 320], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0cd9d17d8c0860f571a917c38828c854(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_08d1fc531e9cffbde81f0f4b433cb9c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 320], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 192, 320], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_72459f194bd6b9967d927180375ec5f5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 3)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 180, 320], dtype='float16'),
            paddle.static.InputSpec(shape=[1, 180, 320], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7169df9e885d852333a98a764c4c95c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72459f194bd6b9967d927180375ec5f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 180, 320], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 180, 320], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0669c93d4506a4a658a3406e8cd92d03(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 720, 1280], dtype='float16'),
            paddle.static.InputSpec(shape=[1, 3, 720, 1280], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5a006f56139e1a2fee6109cae4fe61f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0669c93d4506a4a658a3406e8cd92d03
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 720, 1280], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 3, 720, 1280], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2cea06b6f5afe7b89f6b47f16ac8a303(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[80, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[80, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_acdf880c19faeb73e0cc61dd773ba44c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cea06b6f5afe7b89f6b47f16ac8a303
    def get_inputs(self):
        return [
            paddle.uniform([80, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([80, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b1aa40ecca42422ba28ea17a7353ecef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[40, 40], dtype='float32'),
            paddle.static.InputSpec(shape=[40, 40], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7bdee52edd6927b2bfde6a6e968f7f3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1aa40ecca42422ba28ea17a7353ecef
    def get_inputs(self):
        return [
            paddle.uniform([40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_fc9e66840acad803e10bdb2c9f2bb760(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.stack(input_0, 2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20, 20], dtype='float32'),
            paddle.static.InputSpec(shape=[20, 20], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d721ad6caac645d78b32f1bda4922f39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc9e66840acad803e10bdb2c9f2bb760
    def get_inputs(self):
        return [
            paddle.uniform([20, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([20, 20], dtype='float32', min=0, max=0.5),
        ]


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