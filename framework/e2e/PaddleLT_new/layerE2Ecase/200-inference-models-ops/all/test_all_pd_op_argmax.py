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
class PrimitiveOp_08817e0b27bc06a82c98fd3a78722b68(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1
        return paddle._C_ops.argmax(input_0, input_1, False, False, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_77a5f4818b59c98f11f2f48d06b774f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_08817e0b27bc06a82c98fd3a78722b68
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 512, 512], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c628607bf201aa223089a168cce59752(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_08817e0b27bc06a82c98fd3a78722b68
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 1024, 1024], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4796034e2a778f91df0e66107dbd28f4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = -1
        return paddle._C_ops.argmax(input_0, input_1, False, False, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d636a0224c7c9b3497bf6035018481f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4796034e2a778f91df0e66107dbd28f4
    def get_inputs(self):
        return [
            paddle.uniform([1, 26, 37], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1d606167bc23eb00c7aa602e9d68f3cf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = -1
        return paddle._C_ops.argmax(input_0, input_1, False, False, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_34858c9afd26a3908a2356efce6cd918(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d606167bc23eb00c7aa602e9d68f3cf
    def get_inputs(self):
        return [
            paddle.to_tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype='int32').reshape([1, 26]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_459fbf5ecfa8fdcfdf69f0dd0bab59ff(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1
        return paddle._C_ops.argmax(input_0, input_1, False, False, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_65f5c52ddd1b475ff197f318c7c74af8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_459fbf5ecfa8fdcfdf69f0dd0bab59ff
    def get_inputs(self):
        return [
            paddle.to_tensor([[16.427490234375, 17.15012550354004, 17.204389572143555, 16.34445571899414, 17.16312599182129, 14.864002227783203, 16.76640510559082, 16.61443519592285, 15.000785827636719, 16.2452392578125, 17.437480926513672, 16.663591384887695, 16.742534637451172, 15.812763214111328, 17.00668716430664, 16.412458419799805, 16.183443069458008, 17.512819290161133, 16.917312622070312, 16.713476181030273, 15.516624450683594, 16.228954315185547, 16.265607833862305, 15.796575546264648, 16.383405685424805, 17.374767303466797, 15.152314186096191, 17.792070388793945, 16.29103660583496, 15.86535930633545]], dtype='float32').reshape([1, 30]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_288b94ab78a07620537c94f8040ca150(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1
        return paddle._C_ops.argmax(input_0, input_1, False, False, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c6ad68b42e0a7fb205e049decb641ee5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_288b94ab78a07620537c94f8040ca150
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 1024, 2048], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_af24c65e6d8a11d0b6d3fa094bfc7200(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1
        return paddle._C_ops.argmax(input_0, input_1, False, False, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8aa7eb7e5d11aa8ba2b647ea61e36961(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af24c65e6d8a11d0b6d3fa094bfc7200
    def get_inputs(self):
        return [
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ffa48457216b51b0e22500b15a4dbd02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4796034e2a778f91df0e66107dbd28f4
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 768], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_78c578e205638ecbcb08bc231094e1e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_288b94ab78a07620537c94f8040ca150
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 1024, 1024], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_65afec7c6d314a38d566e0be5ef1c307(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = -1
        return paddle._C_ops.argmax(input_0, input_1, False, False, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dbe233b8aba20aeec1eaa1b9f46272f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_65afec7c6d314a38d566e0be5ef1c307
    def get_inputs(self):
        return [
            paddle.uniform([1, 99], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cbaace722d288056aef778cb3862f939(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af24c65e6d8a11d0b6d3fa094bfc7200
    def get_inputs(self):
        return [
            paddle.uniform([1, 70], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1f24f82813c4ad0c0f90f6760b4a6168(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_08817e0b27bc06a82c98fd3a78722b68
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 512, 1024], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6633d8d9c6b0f2d127b938196e84b8db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_08817e0b27bc06a82c98fd3a78722b68
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 224, 398], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c6dae27ea4f270bab519325e9d6059ec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1
        return paddle._C_ops.argmax(input_0, input_1, False, False, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8b6dec7f9b12c38a643e688ce17d3e22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6dae27ea4f270bab519325e9d6059ec
    def get_inputs(self):
        return [
            paddle.uniform([1, 38], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f75abfb80e4f5632f183e9fc66c25b74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_08817e0b27bc06a82c98fd3a78722b68
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 1024, 512], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_218d55028c2308688c825edceb4907ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_08817e0b27bc06a82c98fd3a78722b68
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 1024, 2048], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_46f2c036946045d82fbad148d80bafac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_288b94ab78a07620537c94f8040ca150
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 512, 1024], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1853fc537fc584fa7a56cc855d00237d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_288b94ab78a07620537c94f8040ca150
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 1024, 512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5bcf5b586384aa22b31293c5168ebceb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1
        return paddle._C_ops.argmax(input_0, input_1, False, False, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e6b9e5b4537843adc02b20c2efaa9f79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bcf5b586384aa22b31293c5168ebceb
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0679931640625, 0.3671875, 0.249267578125, 0.266357421875, 0.4169921875, 0.1890869140625, 0.225341796875, 0.09454345703125, 0.2344970703125, 0.32177734375, 0.39306640625, 0.2347412109375, 0.3232421875, 0.08154296875, 0.328125, 0.4677734375, 0.43798828125, 0.3359375, 0.441162109375, 0.132568359375, 0.474853515625, 0.37841796875, 0.42236328125, 0.31787109375, 0.373779296875, 0.46875, 0.45751953125, 0.35205078125, 0.474853515625, 0.30078125]], dtype='float16').reshape([1, 30]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0e643f2658e76bf4cbbbc7895e55656d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af24c65e6d8a11d0b6d3fa094bfc7200
    def get_inputs(self):
        return [
            paddle.uniform([1, 38], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_22ed2c961d428398261be660f3451854(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6dae27ea4f270bab519325e9d6059ec
    def get_inputs(self):
        return [
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a37248d8f6a2617986186c574b6e299a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_288b94ab78a07620537c94f8040ca150
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 512, 512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e82025c4c370861d76eda4e87dd418ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_459fbf5ecfa8fdcfdf69f0dd0bab59ff
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.482073038816452, 0.44309860467910767, 0.4075786769390106, 0.17388765513896942, 0.3064459264278412, 0.18623974919319153, 0.0950276255607605, 0.44690969586372375, 0.21099337935447693, 0.27662697434425354, 0.33469724655151367, 0.3978627324104309, 0.21882104873657227, 0.31885290145874023, 0.24710576236248016, 0.05025278404355049, 0.08640434592962265, 0.47566288709640503, 0.42425745725631714, 0.06979433447122574, 0.4407351315021515, 0.07119720429182053, 0.27782124280929565, 0.2005079984664917, 0.32010647654533386, 0.43453723192214966, 0.01571422815322876, 0.29954156279563904, 0.1202937513589859, 0.022990459576249123]], dtype='float32').reshape([1, 30]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_aafd37d8da8acb2fedb8147988f3eaed(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = -1
        return paddle._C_ops.argmax(input_0, input_1, False, False, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c9e00913fb39a3a450511eab098ac8e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aafd37d8da8acb2fedb8147988f3eaed
    def get_inputs(self):
        return [
            paddle.uniform([1, 99], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_436388ba952c89567efa2f8e7618dc07(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = -1
        return paddle._C_ops.argmax(input_0, input_1, False, False, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_79257b93b4d611b03e3dbef7dee5c91a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_436388ba952c89567efa2f8e7618dc07
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_97391b8a60b3d1ce1a6398f8db7c4b4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6dae27ea4f270bab519325e9d6059ec
    def get_inputs(self):
        return [
            paddle.uniform([1, 70], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dd6d2769b936c711f87acd4638838129(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_436388ba952c89567efa2f8e7618dc07
    def get_inputs(self):
        return [
            paddle.uniform([1, 26, 37], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1fe6f72f13bc2b4f3886b84a35f44296(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bcf5b586384aa22b31293c5168ebceb
    def get_inputs(self):
        return [
            paddle.to_tensor([[16.359375, 14.7265625, 14.375, 15.9296875, 16.46875, 16.078125, 14.8359375, 15.8203125, 16.890625, 15.3828125, 15.8203125, 16.171875, 16.40625, 14.875, 15.9140625, 16.109375, 15.484375, 15.6640625, 15.8125, 16.140625, 15.6640625, 15.1484375, 15.6875, 16.234375, 16.21875, 15.3359375, 16.4375, 15.1484375, 15.265625, 14.4921875]], dtype='float16').reshape([1, 30]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4de020b7cc518b18757155179a431095(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_288b94ab78a07620537c94f8040ca150
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 224, 398], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_45eb21f452a13bd941b6f08730889a87(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1
        return paddle._C_ops.argmax(input_0, input_1, False, False, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2cdcfebb96c4e6277e40a6d4f5ce9344(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45eb21f452a13bd941b6f08730889a87
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 512, 512], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_7673a0fe8acdce3e2854fcb640d807e4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1
        return paddle._C_ops.argmax(input_0, input_1, False, False, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 19, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f383a0d6868beab9e63a66a4481a22a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7673a0fe8acdce3e2854fcb640d807e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 1024, 1024], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_64228cff3c6dea709d4a01e32fac0541(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = -1
        return paddle._C_ops.argmax(input_0, input_1, False, False, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 26, 37], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cc4718385dfa8f0058688dce1216cc92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_64228cff3c6dea709d4a01e32fac0541
    def get_inputs(self):
        return [
            paddle.uniform([1, 26, 37], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_feb8e5904431a2251af6ec422bd53b29(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = -1
        return paddle._C_ops.argmax(input_0, input_1, False, False, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 26], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_98cba046859df330f08bb231484b5aa4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_feb8e5904431a2251af6ec422bd53b29
    def get_inputs(self):
        return [
            paddle.to_tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype='int32').reshape([1, 26]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c26ebf9b86d82356ca11ec78f54b1dc2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1
        return paddle._C_ops.argmax(input_0, input_1, False, False, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 30], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9618c8a384c73a9eb46e1ba68fb4a960(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c26ebf9b86d82356ca11ec78f54b1dc2
    def get_inputs(self):
        return [
            paddle.to_tensor([[16.427490234375, 17.15012550354004, 17.204389572143555, 16.34445571899414, 17.16312599182129, 14.864002227783203, 16.76640510559082, 16.61443519592285, 15.000785827636719, 16.2452392578125, 17.437480926513672, 16.663591384887695, 16.742534637451172, 15.812763214111328, 17.00668716430664, 16.412458419799805, 16.183443069458008, 17.512819290161133, 16.917312622070312, 16.713476181030273, 15.516624450683594, 16.228954315185547, 16.265607833862305, 15.796575546264648, 16.383405685424805, 17.374767303466797, 15.152314186096191, 17.792070388793945, 16.29103660583496, 15.86535930633545]], dtype='float32').reshape([1, 30]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_433b2a3d3bfa09a79897bc79e1c2e714(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1
        return paddle._C_ops.argmax(input_0, input_1, False, False, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 19, 1024, 2048], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fdfe42f9ed1e2a56369dee20a9f3f5d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_433b2a3d3bfa09a79897bc79e1c2e714
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 1024, 2048], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5a784091ba723efc93a7ecadb21c3bc7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1
        return paddle._C_ops.argmax(input_0, input_1, False, False, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 92], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fd2a302cbcea5cccfb045e07d93cc3d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a784091ba723efc93a7ecadb21c3bc7
    def get_inputs(self):
        return [
            paddle.uniform([1, 92], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d41ab5a95004051be0f9132cce04fe18(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = -1
        return paddle._C_ops.argmax(input_0, input_1, False, False, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 17, 768], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7dcb6c41625c29a66e884ba8b2a76a1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d41ab5a95004051be0f9132cce04fe18
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 768], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_501e9a9a6a559d8127a21c4a87637bea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1
        return paddle._C_ops.argmax(input_0, input_1, False, False, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 19, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8039b39700d7e66d7531b3535d089d70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_501e9a9a6a559d8127a21c4a87637bea
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 1024, 1024], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_63914e1455d6b17c59252072f640e1c3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = -1
        return paddle._C_ops.argmax(input_0, input_1, False, False, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 99], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_756ec9ea4dd6c345c74461be70cd7786(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63914e1455d6b17c59252072f640e1c3
    def get_inputs(self):
        return [
            paddle.uniform([1, 99], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_239a9a6a1ff7acc1152c765c6c44d19b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1
        return paddle._C_ops.argmax(input_0, input_1, False, False, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 70], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_44386907ea09693df55c6f047c73806e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_239a9a6a1ff7acc1152c765c6c44d19b
    def get_inputs(self):
        return [
            paddle.uniform([1, 70], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_caedf9a7e50829b73bd72e20a177f077(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7673a0fe8acdce3e2854fcb640d807e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 512, 1024], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0727321616850800e52b721e6dde0f51(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1
        return paddle._C_ops.argmax(input_0, input_1, False, False, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 19, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9ae3d8ee5b9c7bb3b764c744a64bbd3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0727321616850800e52b721e6dde0f51
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 1024, 1024], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_538df6ce922c79e8c1ea9cc9fa4b336b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45eb21f452a13bd941b6f08730889a87
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 224, 398], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_00ee2d6f809a0e541b98728fa7a04a92(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1
        return paddle._C_ops.argmax(input_0, input_1, False, False, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 38], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b27a4064ad135322ad70084a16379dae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_00ee2d6f809a0e541b98728fa7a04a92
    def get_inputs(self):
        return [
            paddle.uniform([1, 38], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_927df19146be5e0440e412896545dfbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7673a0fe8acdce3e2854fcb640d807e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 1024, 512], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_56d972895d0186c5c5ec568394006a45(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1
        return paddle._C_ops.argmax(input_0, input_1, False, False, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 19, 1024, 2048], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d370c4addbbd87fe676db662cbad7616(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56d972895d0186c5c5ec568394006a45
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 1024, 2048], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_00a8e1e2653a5579b239c246b8a3ddc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_501e9a9a6a559d8127a21c4a87637bea
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 512, 1024], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8cce7ad689bbbf6d7b7153b5ab3fe815(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_501e9a9a6a559d8127a21c4a87637bea
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 1024, 512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0a59d5b3a987dbbe1e697eb27bfccd35(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1
        return paddle._C_ops.argmax(input_0, input_1, False, False, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 30], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ffefb8c3368ed7e3b2598879679c6b9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a59d5b3a987dbbe1e697eb27bfccd35
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0679931640625, 0.3671875, 0.249267578125, 0.266357421875, 0.4169921875, 0.1890869140625, 0.225341796875, 0.09454345703125, 0.2344970703125, 0.32177734375, 0.39306640625, 0.2347412109375, 0.3232421875, 0.08154296875, 0.328125, 0.4677734375, 0.43798828125, 0.3359375, 0.441162109375, 0.132568359375, 0.474853515625, 0.37841796875, 0.42236328125, 0.31787109375, 0.373779296875, 0.46875, 0.45751953125, 0.35205078125, 0.474853515625, 0.30078125]], dtype='float16').reshape([1, 30]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5606dee82565f23fd6074cfc8b980267(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1
        return paddle._C_ops.argmax(input_0, input_1, False, False, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 38], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_89b363fa0cb0a406c5237682ed03bdf0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5606dee82565f23fd6074cfc8b980267
    def get_inputs(self):
        return [
            paddle.uniform([1, 38], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b85aad313ed90a7c18ef4418bbbf50d3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1
        return paddle._C_ops.argmax(input_0, input_1, False, False, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 92], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_58118a245d8d82cad8e8c67ef40f66f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b85aad313ed90a7c18ef4418bbbf50d3
    def get_inputs(self):
        return [
            paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8430a6f9683da8057b7da2757d7d80c2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1
        return paddle._C_ops.argmax(input_0, input_1, False, False, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_114efa79697f17a61bcb3f748be962fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8430a6f9683da8057b7da2757d7d80c2
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 512, 512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f255fe890424cc480421c759219cdc3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7673a0fe8acdce3e2854fcb640d807e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 1024, 2048], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fdb8e49031c3a5ca22a6edb0c5b537b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c26ebf9b86d82356ca11ec78f54b1dc2
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.482073038816452, 0.44309860467910767, 0.4075786769390106, 0.17388765513896942, 0.3064459264278412, 0.18623974919319153, 0.0950276255607605, 0.44690969586372375, 0.21099337935447693, 0.27662697434425354, 0.33469724655151367, 0.3978627324104309, 0.21882104873657227, 0.31885290145874023, 0.24710576236248016, 0.05025278404355049, 0.08640434592962265, 0.47566288709640503, 0.42425745725631714, 0.06979433447122574, 0.4407351315021515, 0.07119720429182053, 0.27782124280929565, 0.2005079984664917, 0.32010647654533386, 0.43453723192214966, 0.01571422815322876, 0.29954156279563904, 0.1202937513589859, 0.022990459576249123]], dtype='float32').reshape([1, 30]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_388e1c863bdccccd181eff1cc4a1fc27(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = -1
        return paddle._C_ops.argmax(input_0, input_1, False, False, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 99], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f6b4ec50e0f5a373e47a558cdaa5cbf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_388e1c863bdccccd181eff1cc4a1fc27
    def get_inputs(self):
        return [
            paddle.uniform([1, 99], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_33bc0022e8aa98be56b3baed0bbf5a03(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1
        return paddle._C_ops.argmax(input_0, input_1, False, False, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 19, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_13bd4e8785683eb160a712b72b6ac14b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_33bc0022e8aa98be56b3baed0bbf5a03
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 1024, 1024], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6827ea97866114e1960f426235de917f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = -1
        return paddle._C_ops.argmax(input_0, input_1, False, False, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 17, 768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ceb75ccfe99b4e9b3777cca7703c7c10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6827ea97866114e1960f426235de917f
    def get_inputs(self):
        return [
            paddle.uniform([1, 17, 768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4f4c95b7643ea8820b037a56c0d4efab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = 1
        return paddle._C_ops.argmax(input_0, input_1, False, False, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 70], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_38921a9401e249da4e358546d86fe58f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f4c95b7643ea8820b037a56c0d4efab
    def get_inputs(self):
        return [
            paddle.uniform([1, 70], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_12b68442120291c1bef077190970cb6c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = -1
        return paddle._C_ops.argmax(input_0, input_1, False, False, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 26, 37], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8a51a5b12f14d78e0fa116ee3a62cabc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12b68442120291c1bef077190970cb6c
    def get_inputs(self):
        return [
            paddle.uniform([1, 26, 37], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_42429ed86eb403dbaa2ce91163818943(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a59d5b3a987dbbe1e697eb27bfccd35
    def get_inputs(self):
        return [
            paddle.to_tensor([[16.359375, 14.7265625, 14.375, 15.9296875, 16.46875, 16.078125, 14.8359375, 15.8203125, 16.890625, 15.3828125, 15.8203125, 16.171875, 16.40625, 14.875, 15.9140625, 16.109375, 15.484375, 15.6640625, 15.8125, 16.140625, 15.6640625, 15.1484375, 15.6875, 16.234375, 16.21875, 15.3359375, 16.4375, 15.1484375, 15.265625, 14.4921875]], dtype='float16').reshape([1, 30]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5c1e2d2a0a3372b54f5273777986b1f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_501e9a9a6a559d8127a21c4a87637bea
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 1024, 2048], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e6bbce370bdc9b54b3a8a24222d2e7f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8430a6f9683da8057b7da2757d7d80c2
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 224, 398], dtype='float32', min=0, max=0.5),
        ]


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