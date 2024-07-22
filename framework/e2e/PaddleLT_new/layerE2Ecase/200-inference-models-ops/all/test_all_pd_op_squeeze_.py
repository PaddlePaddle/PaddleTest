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
class PrimitiveOp_adaa01dcc3930cd4b81d1821b711d762(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9348a555a6215fbcd6a785743f73d87c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adaa01dcc3930cd4b81d1821b711d762
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_28ad7095dac2102156374c651247b590(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adaa01dcc3930cd4b81d1821b711d762
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0875fc82ec969517718d18fefe96084a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d7ba0957489fe7d172c71c4957df715a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0875fc82ec969517718d18fefe96084a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 1, 262, 262], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a100cd42715a73153bb02e1db763473b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0875fc82ec969517718d18fefe96084a
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 66, 66], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_490e8485a24b8fa4f55d9e24411ec67a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0875fc82ec969517718d18fefe96084a
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 262, 262], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c71bba8cc56e2aba7680a5da227da9cd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2, 3], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bdfdcb105d9f9bfee100cc0321c0e9e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c71bba8cc56e2aba7680a5da227da9cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d6ca2d287efd992f5fc8c78b249840e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c71bba8cc56e2aba7680a5da227da9cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4c498e98699d9ba8e96702de116ae828(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c71bba8cc56e2aba7680a5da227da9cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_277da9d1442afda8893533c35b70b62e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c71bba8cc56e2aba7680a5da227da9cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_386813df41c73cf5ca7f909e447956eb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2, 3], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0bba3494b3e4eb10ef101e6bdb71ecce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_386813df41c73cf5ca7f909e447956eb
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5edf2249afc3f39066d064ef442843ce(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([-2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_792f9e41b8ab6ffd3b1d072a40523e51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5edf2249afc3f39066d064ef442843ce
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 1, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([-2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1fcfdf2611177207b016e26787c8d675(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5edf2249afc3f39066d064ef442843ce
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([-2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e0dc460aea51f25bd16272a725c8de49(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([-2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8fb2f42c82dcb6d0bf4742b48e92a34d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0dc460aea51f25bd16272a725c8de49
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 1, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_688da0f6440733f2839e2a0ba91bba2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0dc460aea51f25bd16272a725c8de49
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_23b2e79c22c8d57c7c369c26df24bc28(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_905abf483f6de06790d0434a754ba136(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23b2e79c22c8d57c7c369c26df24bc28
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 576], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4384122efe41e8ad76c1b2a1625904cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_386813df41c73cf5ca7f909e447956eb
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_213e4b987448082d8c3616cc8ca27599(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0026453075a2238e27d9966895655ca6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_213e4b987448082d8c3616cc8ca27599
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 576], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eea89d96bc73b2aeccf591a2f2415624(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adaa01dcc3930cd4b81d1821b711d762
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 25], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f0ece192fffcf941a554a7c0f570ace6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_57ea884925efdc3e6ffbf35eb4a3514b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0ece192fffcf941a554a7c0f570ace6
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 25], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_abc8d82f37daa251b7dc50477098571a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_386813df41c73cf5ca7f909e447956eb
    def get_inputs(self):
        return [
            paddle.uniform([1, 1280, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_15f45939676851b7b6c898c3b0645c7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_213e4b987448082d8c3616cc8ca27599
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_853b67b4c3da4d48736bf448851e546e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adaa01dcc3930cd4b81d1821b711d762
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6c84c58f251dbfbef5b166c8df73a902(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c71bba8cc56e2aba7680a5da227da9cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_14815d8f680e250edc267816bb527264(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adaa01dcc3930cd4b81d1821b711d762
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a04276e9f202c624c208c96ef321410d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4642a34d3bcf6d023dce7f921df86877(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a04276e9f202c624c208c96ef321410d
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9283ab167f8268a00bc7eb6d7622cc98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adaa01dcc3930cd4b81d1821b711d762
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 25], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4630d435183a3bd386c853e3f124452e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_213e4b987448082d8c3616cc8ca27599
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9fa5f3c0b6924f5fb8aa63b97d97515a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0dc460aea51f25bd16272a725c8de49
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8758bf5af469c1dbc5969655e0bde109(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0dc460aea51f25bd16272a725c8de49
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 1, 100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bbd22d5349795e3a9bad51772c3378a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_213e4b987448082d8c3616cc8ca27599
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8ababb702ea633a639eb4b8881067de7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e0dc460aea51f25bd16272a725c8de49
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7f10ff4792d16f7fc6e0c9d44ad119fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0875fc82ec969517718d18fefe96084a
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 100, 100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_940d30cb12484c21103694f0534b18f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_386813df41c73cf5ca7f909e447956eb
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9f3b9ff622efd2cef886c42f58b3facb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_386813df41c73cf5ca7f909e447956eb
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a2a715c55516792ab073cc8c6dfeacea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_386813df41c73cf5ca7f909e447956eb
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_3e3f564e41cb1515ead3af7bb1dced59(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2cedf3b5d15b188e586a931a199750e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e3f564e41cb1515ead3af7bb1dced59
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 400, 4], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b1d23ff91f2f9f647f3734eda6ecbf11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e3f564e41cb1515ead3af7bb1dced59
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 1600, 4], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0a9821e6a3590708a10235dd80c2ddc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e3f564e41cb1515ead3af7bb1dced59
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 6400, 4], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_69bacfff6e85f38924ddf816c9468a4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0ece192fffcf941a554a7c0f570ace6
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 49], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c46f70ff4294d656081230522c95373d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0ece192fffcf941a554a7c0f570ace6
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 49], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5bcc86b3bb537a8b6e695251cd819db0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_485dbfad3e9aa949b00ce061e2e940fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bcc86b3bb537a8b6e695251cd819db0
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 1, 262, 262], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d76768360a2e5da743205dbb64b8f85a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bcc86b3bb537a8b6e695251cd819db0
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 66, 66], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1dad1239b073f23ed872fc43cda7806d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bcc86b3bb537a8b6e695251cd819db0
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 262, 262], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_de0ab360dfa09811b032c6321d82fce0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0ece192fffcf941a554a7c0f570ace6
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_92f5ceda49d70e4f924d05eb243a6f90(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5c4e29880f41b80fc6f410abc04ee068(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92f5ceda49d70e4f924d05eb243a6f90
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e3d22e25071bfa5e2a81443f18d7a74f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0ece192fffcf941a554a7c0f570ace6
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 26], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a321726786b8fea6fc6ac4f7a2576dae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23b2e79c22c8d57c7c369c26df24bc28
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_bdb21a641e5dea5959e736bfefcb8a59(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([-1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2eaced0bf1c69a6449bf136f28b53879(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bdb21a641e5dea5959e736bfefcb8a59
    def get_inputs(self):
        return [
            paddle.uniform([1, 1000, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_cf3af4e65461ff23abcf28993bbf61cb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([-1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1b20b8ed4da3d94dc5d85c03a371c380(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf3af4e65461ff23abcf28993bbf61cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 1000, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ee86471f21addec1bc7d1f21521dcd38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5edf2249afc3f39066d064ef442843ce
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 100], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([-2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a697dfeae03ef611d3c4795b41d52d5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5edf2249afc3f39066d064ef442843ce
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 1, 100], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([-2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_350acce586797f6cae2744bb322549f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23b2e79c22c8d57c7c369c26df24bc28
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 100], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b9ffa2e9949a154e574f90d4564270bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5edf2249afc3f39066d064ef442843ce
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 100], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([-2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e11dd0146b9af41de6f798cb11d04444(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bcc86b3bb537a8b6e695251cd819db0
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 100, 100], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d930acd953de85d12b8e0fe6d7658a01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0ece192fffcf941a554a7c0f570ace6
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7db7c6f4cd6c73e942f8257a9a5eae0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0ece192fffcf941a554a7c0f570ace6
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9493417890c1d8e4542f9b5e27d7143f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92f5ceda49d70e4f924d05eb243a6f90
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_85a09b9ee33269874f0b0be4ad957dce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0ece192fffcf941a554a7c0f570ace6
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 25], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6373b98d68b24b6bb754d1591fb1fb69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23b2e79c22c8d57c7c369c26df24bc28
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 192], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4a716bbcaff3bdf3c8b2c669ba5f03a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c71bba8cc56e2aba7680a5da227da9cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1d901107d56e15e90d9e549d79069f4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_386813df41c73cf5ca7f909e447956eb
    def get_inputs(self):
        return [
            paddle.uniform([1, 1000, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a9116085d5f519319a5903a53a75f7d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adaa01dcc3930cd4b81d1821b711d762
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d01bdb221da45bcef58cbab44fedc43e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a04276e9f202c624c208c96ef321410d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ba46e00b94b59b129e69f97fa5a9a593(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adaa01dcc3930cd4b81d1821b711d762
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 26], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_23f236e604038812ff66d3e42fff8942(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_213e4b987448082d8c3616cc8ca27599
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1460e9962373c5f6810b10dc9b3e8ae8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3ee970329264d5a861f10fa9ace268fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1460e9962373c5f6810b10dc9b3e8ae8
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 400, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ad3b35cb0036a1af5fa57e061a7ff768(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1460e9962373c5f6810b10dc9b3e8ae8
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 1600, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dcb4af1e94e8237507c52b052bed9f97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1460e9962373c5f6810b10dc9b3e8ae8
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 6400, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8fa9a2fe651797dd3d69d42105b5c26c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([-1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5c05a806aade47503e9daed4ff590863(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8fa9a2fe651797dd3d69d42105b5c26c
    def get_inputs(self):
        return [
            paddle.uniform([1, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2d2b4e4e1495ec405c54ba9a35d62aa0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([-1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ff137e3ff58a50948a4690f9aa1de12a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d2b4e4e1495ec405c54ba9a35d62aa0
    def get_inputs(self):
        return [
            paddle.uniform([1, 1000, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_70a52edcb4b329e420a21413eaefcefa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c71bba8cc56e2aba7680a5da227da9cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 1280, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7ebae57431ea21ee1e12ee0c23024062(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_23b2e79c22c8d57c7c369c26df24bc28
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 96], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0c04a2e74c11ad58d6648a168aca0fcc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, 1, 49], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_06678af1eebd0ec44a9d84a64656b167(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c04a2e74c11ad58d6648a168aca0fcc
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f7a38cc3b8ec8726ded2543b5e5a86f2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 384, 1, 49], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b404834075c2258b2cfc43322ac8ad19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7a38cc3b8ec8726ded2543b5e5a86f2
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_48220b6ea30f30a3a8630b2f423463fe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 1, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fed53395b5012b7263c4de7265668a98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48220b6ea30f30a3a8630b2f423463fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 1, 262, 262], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b227d004f187daceec2aef4ea75de57c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 1, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_108ead61dd7fefed7b8a527c19d1ac94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b227d004f187daceec2aef4ea75de57c
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 66, 66], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_3eef748a5980f80eca41a6de6fc55b94(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 1, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_375fe84f3101f7c9e6763f37eba0c715(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3eef748a5980f80eca41a6de6fc55b94
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 262, 262], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6f41796b47f59054fa5cee0ca512a4bc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2, 3], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e5ee49af094af409bf8ef1f9ff6d0e2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f41796b47f59054fa5cee0ca512a4bc
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2a8ee721db9212cbfdf0b9f5804b966f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2, 3], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6612bde0776287ad8872cfe615947918(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a8ee721db9212cbfdf0b9f5804b966f
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_37c84a529ea4babd81ec5773454ab299(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2, 3], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1024, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_50550f73b12457995039290f9e0937c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37c84a529ea4babd81ec5773454ab299
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8f3183740af3030daa8bbcc52b967370(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2, 3], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2048, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6d34307b2e4d73f7581fb1ca4ff6bc34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f3183740af3030daa8bbcc52b967370
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9263c1a762dca38dfc46044fc971a8bf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2, 3], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 72, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3f059e08eb3f2ae6ba3823d23a8410db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9263c1a762dca38dfc46044fc971a8bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ae4a130a478c81be2ba2d7ebd617a601(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([-2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2048, 1, 256], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6b5a852a2e54ebd73a3a1394acb51379(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae4a130a478c81be2ba2d7ebd617a601
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 1, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([-2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5bb13280347536d61122c4ea9cebfc2c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([-2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, 1, 256], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_97c363ecc3fa530bb5b858038ae1fd15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bb13280347536d61122c4ea9cebfc2c
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([-2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_7009de1be1a8af8a90cf549fae1bbf18(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([-2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2048, 1, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e330b065741d3b7aa20f24fbd14098a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7009de1be1a8af8a90cf549fae1bbf18
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 1, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f67de139687cba87c948b1a4e7806a43(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([-2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, 1, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_15f801dd574cef328e9de9f79dc8bc1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f67de139687cba87c948b1a4e7806a43
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ace10d4f9ecb135c071ba83288e63ba7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 576], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_20b79c40a1b1d98d50b3e801970ebb82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ace10d4f9ecb135c071ba83288e63ba7
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 576], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e53890d47828f59f09e9c00728ad0dc8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2, 3], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2048, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5a1d8e04e001f6ccee74fe5e44cbb291(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e53890d47828f59f09e9c00728ad0dc8
    def get_inputs(self):
        return [
            paddle.uniform([1, 2048, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a55f032247bc04b14fec91b9783794b6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 576], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8060bd9c139c381eba584da88068250f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a55f032247bc04b14fec91b9783794b6
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 576], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9c9ba67f5968b19cde2a2a1a83a6e41a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, 1, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_286394042a648446332ebf3cd73375e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c9ba67f5968b19cde2a2a1a83a6e41a
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 25], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_50920f178baf36df1a0a7308aed0ce5c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, 1, None], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_648822a6e0843600125808d3c3290c86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50920f178baf36df1a0a7308aed0ce5c
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 25], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a42968f760726fc6bcf54d57a02e1333(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2, 3], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1280, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_821a16fa528aac4699a82aa525b4df1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a42968f760726fc6bcf54d57a02e1333
    def get_inputs(self):
        return [
            paddle.uniform([1, 1280, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_554efe4ff46d1d020704617b262328ad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b15ed746c6fe89bd96f1fd514c24a5bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_554efe4ff46d1d020704617b262328ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 96], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_042d2305a47ecc90ec0a58689d23cb2f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1024, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7a19c8b8df12a5d84ad0e7734991a05a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_042d2305a47ecc90ec0a58689d23cb2f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_eb399cadf43887c850802a1cd6e5dfc8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2, 3], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1000, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7517a0d768d46cd513edba2482468a9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb399cadf43887c850802a1cd6e5dfc8
    def get_inputs(self):
        return [
            paddle.uniform([1, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8b70003a7aaafaf70ea7bd5f828074e1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1b119547ad58b8358a46c7957c2e8832(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b70003a7aaafaf70ea7bd5f828074e1
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d55929ad9c767d40cdccc2bf470f8b56(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ca0c0d840749a7a7f85ed279dcc69868(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d55929ad9c767d40cdccc2bf470f8b56
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6ae2fe6bd36941164e66b3468c46c8fb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 480, 1, 25], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1951f1488d0a7f34da1e805de55427e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ae2fe6bd36941164e66b3468c46c8fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 25], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_94ed2db08c33101f8d0ce7925115a81c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_81ba42d85fa268694a773fd57feb4207(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_94ed2db08c33101f8d0ce7925115a81c
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 192], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4ef86d52c7c7c8641ee9994fee2d56b5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([-2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 1, 100], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cebdd0d4000da8c4b3665eac656d6459(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ef86d52c7c7c8641ee9994fee2d56b5
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_698e73a0a1ce764a568d2b74050f0ff1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([-2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 1, 100], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6be13484c013080fdc19d0a78bfd2aba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_698e73a0a1ce764a568d2b74050f0ff1
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 1, 100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b3cf0d55885e7c03ed73eb39d1a70e41(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 100], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_783e1ca8575985dc5bbc18c98af0d1ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3cf0d55885e7c03ed73eb39d1a70e41
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_529ba92cbb83098dfc1a862302e2812f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([-2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 1, 100], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b01cce759811da4cbbc41d5108989566(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_529ba92cbb83098dfc1a862302e2812f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4f5111c22d6a3995cd32e2b2507eca05(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, None, 100, 100], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0f88bd2811513c63012130dfe86c1098(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f5111c22d6a3995cd32e2b2507eca05
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 100, 100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_417ee55ea478a1fb386f5c241275780e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2, 3], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8494f8b6b0d93c5fbbe1056b913066f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_417ee55ea478a1fb386f5c241275780e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1529e9380c64ed4342f34549728a22d2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2, 3], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4b22fd7af14917cde944d10780c59abb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1529e9380c64ed4342f34549728a22d2
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6729e354f91d4d0d0bd665077cc22c47(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2, 3], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1024, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_82b4eb931c71586fd84cefc72cac41c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6729e354f91d4d0d0bd665077cc22c47
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_aa85631b93327650a22bddf54f4cc898(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 400, 4], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2b5c5b82eaacc82c5ca77c244f1e18ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa85631b93327650a22bddf54f4cc898
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 400, 4], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d882db3dc85af53f17685ccb7530da49(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 1600, 4], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_70a6ff6a77a056219ffca1dc81d74b93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d882db3dc85af53f17685ccb7530da49
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 1600, 4], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_7ada762415187e33138f8e04ab2aff78(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 6400, 4], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2b693a99fbf77bdb651008a66b4bf294(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ada762415187e33138f8e04ab2aff78
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 6400, 4], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_42acf5824d9a883ff9957caaa958c582(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, 1, 49], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5e4379a5ff0a864681f2126e182cc00b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_42acf5824d9a883ff9957caaa958c582
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 49], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_fc28cfdb0ffce6f3cbb240d162fde647(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 384, 1, 49], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f0d5c5ddcffe695118dfaa6d020ce26a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc28cfdb0ffce6f3cbb240d162fde647
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 49], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_46eeab857a67b8ed9e22c32e26544cc1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 1, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_09d1c596e914dca5da76d6b1d1d85d4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46eeab857a67b8ed9e22c32e26544cc1
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 1, 262, 262], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_fec9b566ac9b55b3ec46f501d01574ff(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 1, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fa47191e5aca4631a68b48c0bb276b7b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fec9b566ac9b55b3ec46f501d01574ff
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 66, 66], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ee4484cee3b0e194949bb9f238820401(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 1, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_da4b49efd22b55acaeeb343ea152c239(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ee4484cee3b0e194949bb9f238820401
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 262, 262], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8101f75856928405f3d57222a802384d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_277b56217e08ad692d2b4532847337e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8101f75856928405f3d57222a802384d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_70dec946d4f9069d5d575f570b9eda14(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_07efb969bb7e145baf3686b538a87438(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70dec946d4f9069d5d575f570b9eda14
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f25ec85db60142d97c3421459f5c8fc6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, 1, 26], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c0fff2949b3d8cf1fc42b65a3f088ca2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f25ec85db60142d97c3421459f5c8fc6
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 26], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1e5416ae2ac7d0f820da5651f044752d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 512], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_67f7d0e320ba9f370f7b83861523cd04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e5416ae2ac7d0f820da5651f044752d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_38e8505ca5ec78c77febe4755baf1dda(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([-1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1000, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8bce894af60edd1e862330bb69174b7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38e8505ca5ec78c77febe4755baf1dda
    def get_inputs(self):
        return [
            paddle.uniform([1, 1000, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_46b793550692d6dbc34e5e23ba833a68(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([-1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1000, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c520a70d80f9b823493158abd77515a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46b793550692d6dbc34e5e23ba833a68
    def get_inputs(self):
        return [
            paddle.uniform([1, 1000, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b384762e98777d3d7badc7c859e6b504(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([-2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 1, 100], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ef4b14b57dc279dd892cc638c013c4b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b384762e98777d3d7badc7c859e6b504
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 100], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([-2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a2b49912997f069cdb8a1dca749f3e56(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([-2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 1, 100], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_be4698617f19fef148540699216d0236(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a2b49912997f069cdb8a1dca749f3e56
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 1, 100], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([-2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_cc657f59e525632b706b4723ac3d8a78(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 100], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0eb1e1bc6a29837954e5a5900b282a4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc657f59e525632b706b4723ac3d8a78
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 100], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c51079f311a485d4bcd840568013bc70(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([-2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 1, 100], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_32ddd53267c7751a526a20d07c1d708a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c51079f311a485d4bcd840568013bc70
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 100], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([-2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_df40f708d5b281d4a01a5285fdd0ef3d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, None, 100, 100], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7a431679cbb07a22c16ae88d2c3b3b8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df40f708d5b281d4a01a5285fdd0ef3d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 100, 100], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ebdca3f31cac8b6f1f73efa0e339cc8d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1024, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_176c8fb57c9ab39c9899824f390c8c84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ebdca3f31cac8b6f1f73efa0e339cc8d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8cda117c506031a56ada58d786ceac89(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_453600b7a95ae1e9119efde5f4fdbecb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8cda117c506031a56ada58d786ceac89
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_3d1eb6fbd1955e3279a4b3660707c37b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_24a75bcaeb2546daff5e64345396a203(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d1eb6fbd1955e3279a4b3660707c37b
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e73487094ed1627ee0256f4075df3f3f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 480, 1, 25], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8aac6a9209a1c6e30f9e0e94ba0cf95a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e73487094ed1627ee0256f4075df3f3f
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 1, 25], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_143d434a5bf7d81694338ab705558dd7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 192], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a0826cb691164441d0b40c961d840a71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_143d434a5bf7d81694338ab705558dd7
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 192], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_602eff80f872dbe642f135620596a4d6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2, 3], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 72, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6058c8909086cae0b025bffd3b8bd68d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_602eff80f872dbe642f135620596a4d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f01fe75197ca5186641dee605a48642e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2, 3], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1000, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_161682cd367c7fc82ebe7ff7779d3340(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f01fe75197ca5186641dee605a48642e
    def get_inputs(self):
        return [
            paddle.uniform([1, 1000, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2c43a3e49cc7ae6f574967524209f58d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_60182c20d2b5ba6878b24f57f0767c6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c43a3e49cc7ae6f574967524209f58d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2d7475393b9cb614ac5f2811f15b358a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_50443e13cbc4fe5cd81c0015a67c59e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d7475393b9cb614ac5f2811f15b358a
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_3bc30029bded488c6276964d389a570d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, 1, 26], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3cce57fbc2495d1d47e27e000f09cd12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3bc30029bded488c6276964d389a570d
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 26], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1a7d1ad94846df798835f350544552d2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_82461a2d42288c069c013b345fb45529(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a7d1ad94846df798835f350544552d2
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_3018e6ce3b073b61a68f13e6efc7f6d2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 400, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8ce2056c634d948247aac95f1dd8183a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3018e6ce3b073b61a68f13e6efc7f6d2
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 400, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_11d4e91d4af475b5d080dcef2e4c99b2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 1600, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e62d4c5c9adceafa2df9eb60ab2b73b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11d4e91d4af475b5d080dcef2e4c99b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 1600, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a5524e14ded690b078de7861ded321d5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 6400, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_84ed3ce36ed529ddd16680c8f2405e14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5524e14ded690b078de7861ded321d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 6400, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d8dba382a78af3e9c476867b49120952(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([-1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1000, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2f07fc1e1391af2f9aa0c0fd91c19b5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8dba382a78af3e9c476867b49120952
    def get_inputs(self):
        return [
            paddle.uniform([1, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_dadee71d817bbe4f6db81cd39f0a9218(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([-1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1000, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_84cb91cd1255a31e54ba23811d43c672(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dadee71d817bbe4f6db81cd39f0a9218
    def get_inputs(self):
        return [
            paddle.uniform([1, 1000, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_65ba505ec84309d14825b325241a8672(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2, 3], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1280, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_22c1d266e0d57a65878ac031c0639574(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65ba505ec84309d14825b325241a8672
    def get_inputs(self):
        return [
            paddle.uniform([1, 1280, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_455fb46b7d4bcbf85fcf6b0ef07dfe35(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.squeeze_(input_0, input_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 96], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d821547ad71fdbf296580f0e409fbbc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_455fb46b7d4bcbf85fcf6b0ef07dfe35
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 96], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
        ]


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