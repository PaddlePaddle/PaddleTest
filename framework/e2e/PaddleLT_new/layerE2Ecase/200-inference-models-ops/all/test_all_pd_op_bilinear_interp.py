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
class PrimitiveOp_aaacf496bcebcfc5e506c9c0486d291e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([32, 16], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_31e4b3d6a0872d82ad197a84a3bc1327(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aaacf496bcebcfc5e506c9c0486d291e
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 16], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_454c66933b94b469c784eed6c4a16136(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aaacf496bcebcfc5e506c9c0486d291e
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 2, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 16], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a854e780dfb4c3acef2a326b249c0306(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aaacf496bcebcfc5e506c9c0486d291e
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 4, 4], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 16], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_571a5441d374a6ef7709e61c13bbfdda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aaacf496bcebcfc5e506c9c0486d291e
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 16], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 16], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d45fa4a1ecd7a2f499b88c876dafeb10(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([64, 32], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_986f21971880c21275f446143f24965e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d45fa4a1ecd7a2f499b88c876dafeb10
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 16], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([64, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_339485d1a1b526d94e71b200d7b25502(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([128, 64], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8e02e3fa2f79e8d63d2328e586da12c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_339485d1a1b526d94e71b200d7b25502
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 64, 32], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 64], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4392ac4d187ccd3b1c64321e3467bb30(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1024, 512], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0dedd9c3bce85937cb9c8110be8e3c4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4392ac4d187ccd3b1c64321e3467bb30
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 128, 64], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1024, 512], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f0e12afb3095bff7420400964accf472(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 8, 6, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8e17226663fb734f7860849644b2dd32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0e12afb3095bff7420400964accf472
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 4, 3], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_deea7be378e111959ff5115aba0ceb43(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 16, 12, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_47819d739ca25a9b91516b5c77dc0c0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_deea7be378e111959ff5115aba0ceb43
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 8, 6], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_377a7709b69021f3fa849a54c29eca14(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 32, 24, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4fa8d49cad4a9e6d3e44377b8eb780ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_377a7709b69021f3fa849a54c29eca14
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 16, 12], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_3d8f7fd95961fa2e2e0d95ce7b1cca7e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([128, 128], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6aec3a84b9e448f8a8d89f6a34c2190b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d8f7fd95961fa2e2e0d95ce7b1cca7e
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 64], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ccab8c49af514b2296e3f69f73c0efc8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d8f7fd95961fa2e2e0d95ce7b1cca7e
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 32], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_239f9b2ea3c01cdeb3487d3b7ba7c66e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([32, 32], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d59374ff6298cc94cf06837d4d637d55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_239f9b2ea3c01cdeb3487d3b7ba7c66e
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 16, 16], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a453b6d48028fa332d2c1ceeb64872f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_239f9b2ea3c01cdeb3487d3b7ba7c66e
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 8, 8], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_facd1c9ddab1f5ac102527273cef7b86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_239f9b2ea3c01cdeb3487d3b7ba7c66e
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 4, 4], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8218396e259d68aa70950984a1445580(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_239f9b2ea3c01cdeb3487d3b7ba7c66e
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_908b3774633790ea1a79b08187dd6755(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1024, 1024], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9b41a75a52549d5c08fd83c9f935a007(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_908b3774633790ea1a79b08187dd6755
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 128, 128], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1024, 1024], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_05871afc89d0029902cfe99fdcf47784(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([128, 256], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ec06fe678c657ca90dd66890d84550b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05871afc89d0029902cfe99fdcf47784
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 64, 128], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d298658402b4efede88efe607be5ea6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05871afc89d0029902cfe99fdcf47784
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 32, 64], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1294afd8fa656cae26090af463722add(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([64, 128], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_654f8ed10251f487793a38930543222d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1294afd8fa656cae26090af463722add
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 32, 64], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([64, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_13aa2982e1b7bb450e9d2bdb0504e82d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05871afc89d0029902cfe99fdcf47784
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 16, 32], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_22a44c7d5d71ce668cbd3b18f7a4d458(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1294afd8fa656cae26090af463722add
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 16, 32], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([64, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_63ce8568a3d7ca0f263a1fa6c57840fc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([32, 64], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_246b6f155ed8e7ceca267a64288fdde5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63ce8568a3d7ca0f263a1fa6c57840fc
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 16, 32], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 64], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_df69b69f1d7f6864ba51a5875bb68ddb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05871afc89d0029902cfe99fdcf47784
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 64, 128], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5dd9f16dc837ac5d30673a880b2f67b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05871afc89d0029902cfe99fdcf47784
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 32, 64], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_97c7a1473908a14c96af40f838219ec0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_05871afc89d0029902cfe99fdcf47784
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 16, 32], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c8f044d83c7c46dc9f5bbdf0b6f4ca65(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([512, 1024], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dd76908fe21433f9bd85409e41ec0d2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8f044d83c7c46dc9f5bbdf0b6f4ca65
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([512, 1024], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_764819ce8f99c75f589ecf747878f54f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([112, 199], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a0f438874501cf591d874bd5cf70d2f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_764819ce8f99c75f589ecf747878f54f
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 14, 25], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([112, 199], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_08b53211a05725e967c86a148fd04409(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([224, 398], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f99df17f369649676fcb6517948e7e49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_08b53211a05725e967c86a148fd04409
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 112, 199], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([224, 398], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2dc503238519c16c42b24f1f7b8ae250(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([128, 128], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_27bc9e5d7fa9eaeb3071c81ab21ebc34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2dc503238519c16c42b24f1f7b8ae250
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2347fd5bfe59060dd77d01b888f5d67b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1024, 1024], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_768da052f3cd1e44a3ac71381c52f66c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2347fd5bfe59060dd77d01b888f5d67b
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1024, 1024], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4e29342361cd1ede52b02d131ca5910c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([128, 128], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_57b5d414be8f18131062d67430fa7e12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e29342361cd1ede52b02d131ca5910c
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c1db72460f3395e3335c6510b5df1d29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e29342361cd1ede52b02d131ca5910c
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 2, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0da23dbbac314779c89968496d79eae6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e29342361cd1ede52b02d131ca5910c
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 3, 3], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_91321b5b7ea2a3211ddcf335b730802c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e29342361cd1ede52b02d131ca5910c
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 6, 6], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dd569fe203d70dac9a6176f11444f7e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e29342361cd1ede52b02d131ca5910c
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 128], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_71d9c16ec9f726e0918879b26be2054d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([256, 256], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8b9b2c7ce02eda52aa19e51f3dc2f34f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71d9c16ec9f726e0918879b26be2054d
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 128], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([256, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bbc36bae717be670e8b12ecc32c72fcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71d9c16ec9f726e0918879b26be2054d
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 128], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([256, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f82a8cdf3b4d1b5dd09c7d52bda14118(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_908b3774633790ea1a79b08187dd6755
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 256, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1024, 1024], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_85f4cc9e3c1cdc510a2a08fad181b0e4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([128, 128], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d39e05038a7b080ef35966fca0f2d669(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85f4cc9e3c1cdc510a2a08fad181b0e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c64f6267fa38ce466ff6edd36015fb39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85f4cc9e3c1cdc510a2a08fad181b0e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 2, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_25c73288a1dfe260faf768281b842fa4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85f4cc9e3c1cdc510a2a08fad181b0e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a4c00d60f4c3551bc17e49abe167542e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85f4cc9e3c1cdc510a2a08fad181b0e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 6, 6], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8627e906593d892d4c35f30ba64d1cc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85f4cc9e3c1cdc510a2a08fad181b0e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a18ae0d000314790c377bf195d56d012(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([256, 256], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9b60ea71acbb9fca9487b1195d1ab709(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a18ae0d000314790c377bf195d56d012
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([256, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f178eb597a5d03991426360ba277a2e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a18ae0d000314790c377bf195d56d012
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([256, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d7cffcff59fa72a394386920706edf2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2347fd5bfe59060dd77d01b888f5d67b
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 256, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1024, 1024], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_fc0414a22539463f493048611304c1b4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([128, 256], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_21bdaa80ee09d8e0507ad7ac46e27759(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc0414a22539463f493048611304c1b4
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d2eeabc317e7d932493919bfc9d134a8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([256, 512], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fcb84016c87993d1ca5fbb8663d4b6d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2eeabc317e7d932493919bfc9d134a8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([256, 512], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8fffb8d535e6a7b995a99318dff158e6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5ccd5a0a9693c28602003bee1d2b87b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8fffb8d535e6a7b995a99318dff158e6
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3496dc5b8cee8dc723ff9c8a853b6418(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8fffb8d535e6a7b995a99318dff158e6
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 256, 512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c877f5103b84003861f66aa491eb69ac(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([256, 512], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_55a6bd3aaf51fee3d8425ba00acfb50b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c877f5103b84003861f66aa491eb69ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 512, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([256, 512], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5dc6226027210f193d0d668f611c0d5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c877f5103b84003861f66aa491eb69ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([256, 512], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6c96e530ea131772e453982d7b6fe3ff(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1024, 2048], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0a1605d6bf0bc2d212d854f7f33bb4ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6c96e530ea131772e453982d7b6fe3ff
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1024, 2048], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_75c378fc6566d19bcbd69f86fb259a22(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([64, 64], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_92591e07af60eb7379270297231cb4ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75c378fc6566d19bcbd69f86fb259a22
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([64, 64], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0cc8732c810de671807cef0319cfe04f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2dc503238519c16c42b24f1f7b8ae250
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_14d3f47c064842c6180b045d764c13b1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([512, 512], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a74aa278654a40d5f2ac3dd03e18ba79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_14d3f47c064842c6180b045d764c13b1
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([512, 512], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e8c175d775dca880967a0c81840181fc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([112, 199], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8fa317c59244d09410408de1dc280202(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e8c175d775dca880967a0c81840181fc
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 14, 25], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([112, 199], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a750898017eb6aefcc176095461eac41(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([224, 398], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4ae7fa0ba1ea2ac1c8e9c55816950a38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a750898017eb6aefcc176095461eac41
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 112, 199], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([224, 398], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_359fe2d9fc6d61698c791113442cb3ad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 32, 64, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bb358c3830528c1e632c57ffd328d377(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_359fe2d9fc6d61698c791113442cb3ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 64, 256], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a4383b41fb03d3ab7c982a4f398547ae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([32, 32], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c9c2d2d623f1a78f6babfc18a4f3792f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4383b41fb03d3ab7c982a4f398547ae
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1b55795b711167e1370b9ec92e41ca26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4383b41fb03d3ab7c982a4f398547ae
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 2, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_839c9e42b56c57d8b9762a5c8cde14bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4383b41fb03d3ab7c982a4f398547ae
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 3, 3], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_74d47e9c19804c865a20d7463a7a43df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4383b41fb03d3ab7c982a4f398547ae
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 6, 6], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a648457304f2bd1b02c35b01388cf901(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([64, 64], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_77d663b0a197ed91f590ef0106bad3f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a648457304f2bd1b02c35b01388cf901
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([64, 64], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_031152da8e6c293cf55fb19d4c52a570(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d8f7fd95961fa2e2e0d95ce7b1cca7e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_85d4ba1f9f56c65d5170e710a7e24369(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([512, 512], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eed756674b39561b641e77d48b30ea54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85d4ba1f9f56c65d5170e710a7e24369
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([512, 512], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_cc07439b6a312a88f8a105c3c119daa6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([128, 256], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_16bd92e641916c39de113bb68ae5fbf6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc07439b6a312a88f8a105c3c119daa6
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f20f9eb0ba1374e87bbf1b8b35a0a73c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc07439b6a312a88f8a105c3c119daa6
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_09e9099430afb21b5d793fe0974cabc6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([64, 128], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d0af93cc8dec855a9b0c0a40138ff26f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09e9099430afb21b5d793fe0974cabc6
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([64, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_61e3286c7efd0556877e79843ba2c0d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc07439b6a312a88f8a105c3c119daa6
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_751b47bc8b18d50a8093fe36917f647b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09e9099430afb21b5d793fe0974cabc6
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([64, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a62eafd439010d0b1708f785a9ae9609(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([32, 64], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bc96519db24b8dea5786239826b7be1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a62eafd439010d0b1708f785a9ae9609
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 64], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_df2cdde3a1d0c5b8ff49ff2e49a89755(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc07439b6a312a88f8a105c3c119daa6
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_663f9934547b77aaba894761cd901583(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc07439b6a312a88f8a105c3c119daa6
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7cb9ba104a5433140d738e4fe3212bf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc07439b6a312a88f8a105c3c119daa6
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_38d77bc147b8eb7724ef15673fe2f2e6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([512, 1024], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3be12070866cc286a0886c8d0af2a225(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38d77bc147b8eb7724ef15673fe2f2e6
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([512, 1024], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f5789d977234efe7290abce8f3f89e00(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 8, 6, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4e02491c8f4b28747a359454148c460c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5789d977234efe7290abce8f3f89e00
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 4, 3], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d73144cfa0de36853d76314cabc3e262(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 16, 12, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_20eda6e410493335c24685f92c1c195f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d73144cfa0de36853d76314cabc3e262
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 8, 6], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e1cc0356eef3ab8662516fb176c0acf6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 32, 24, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9b97433ac7b65faf7bc38efc6e480482(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1cc0356eef3ab8662516fb176c0acf6
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 16, 12], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_7dad7ccb4d9b23b90085233c0c593062(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 32, 100, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2dd25015eeda298892d309ba5404bd1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7dad7ccb4d9b23b90085233c0c593062
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]]], dtype='float16').reshape([1, 1, 2, 6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5b64cb65fd5a45b210ac36f18e3953fb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 32, 100, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c6e2bb93a444c153c418cbcfc602c83a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5b64cb65fd5a45b210ac36f18e3953fb
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]]], dtype='float32').reshape([1, 1, 2, 6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1a670ff477491551eb14694ff5272358(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([32, 16], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_866b1300756ce83c6d94af716b9d1836(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a670ff477491551eb14694ff5272358
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 16], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_417a3b2608dd2110f2facdf5f59cdea7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a670ff477491551eb14694ff5272358
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 2, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 16], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5cfc9dcaa92fc665010a85842d208f33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a670ff477491551eb14694ff5272358
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 16], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_daef6ff44dd2c7c6b1b02b3e88443d14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a670ff477491551eb14694ff5272358
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 16], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d2c6154d424e46db8158548eee9c567b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([64, 32], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8c3f6b5e1e01a30725e686d5cb1c5749(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2c6154d424e46db8158548eee9c567b
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([64, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_19696f0a499715e4e475c7f08112e276(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([128, 64], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cc61bfbb28c4762e0c42027bf8e0e274(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19696f0a499715e4e475c7f08112e276
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 64, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 64], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8a3352a2d7be4564b175063c38162498(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1024, 512], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_509b2ae0a146d79adb2e6faac746c5c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a3352a2d7be4564b175063c38162498
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 128, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1024, 512], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a5197e748f6f0d7b3064f02b0b413e71(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([32, 32], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c971b565b8b523f8304b9d8f7b05086d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5197e748f6f0d7b3064f02b0b413e71
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0c36c56595143c87ae01c19984c66792(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5197e748f6f0d7b3064f02b0b413e71
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 2, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_66a3f1f5a71b82f2cfd910d8ead2d7f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5197e748f6f0d7b3064f02b0b413e71
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1ff07dbd0f18f7e031a0519f09c49bf1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5197e748f6f0d7b3064f02b0b413e71
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 6, 6], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b5edae41fa5789890046f31170e3a72a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 32, 64, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_49839d4eff4229867c645bd446916b01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b5edae41fa5789890046f31170e3a72a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 64, 256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_157ea28316a03a7de9e0385350fc9b00(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 192, 320, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dfa09ac2a2eec1782effbc08284fb111(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_157ea28316a03a7de9e0385350fc9b00
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 180, 320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d71791b8574b3ece096725031ee416d6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8cc4b0e206de335a2e6e0bfbf3f5a683(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d71791b8574b3ece096725031ee416d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 48, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b62a5d21e9c4d444cdae54558cfce87f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d71791b8574b3ece096725031ee416d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 96, 160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_fc67b0940969029d5b1b939acf35c52e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 180, 320, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_705f6694b8202c0728b9afaead71f572(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc67b0940969029d5b1b939acf35c52e
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 192, 320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_bbf1cde46fe1b3e37979c812068af0df(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [float('4'), float('4')], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ff003d32314851e01b71356309cc7e06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bbf1cde46fe1b3e37979c812068af0df
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 180, 320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0a61863c4cd72689aadf31b99f8a68cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2dc503238519c16c42b24f1f7b8ae250
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f4a251c89c03757df5a932683c969da6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([32, 32], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_27ddee8ec6526772b933fe54e68214a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a251c89c03757df5a932683c969da6
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8b3c202c732c2f14197906b1fd29a230(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a251c89c03757df5a932683c969da6
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bae1607be001ed3e442cb988fbb1e588(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a251c89c03757df5a932683c969da6
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fb3b1b04ef97ca63719f76e8a74b97d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4a251c89c03757df5a932683c969da6
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c3dc931343004c4ca9c529f161c403f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([128, 256], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bebdd5efe79b3e624f0eeb8845ab9218(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3dc931343004c4ca9c529f161c403f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ce77e94b4bb7b26c3064a3ac2741f142(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([256, 512], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b2df8eefe311dfebab567ed34adc02f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce77e94b4bb7b26c3064a3ac2741f142
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([256, 512], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_71666d6dd34cf0c9a3268e1c2748e865(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_19e79e96734c83568a98bd6c3f28f44d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71666d6dd34cf0c9a3268e1c2748e865
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 256], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2a9617909d3c5fcb859e577a9767c194(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71666d6dd34cf0c9a3268e1c2748e865
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 256, 512], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c1f1b64cc24e8a528f06e088d59beabc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([256, 512], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8e063bd0c1030a6f72a98956f59272db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1f1b64cc24e8a528f06e088d59beabc
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 512, 1024], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([256, 512], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_14e57f1f4a0727e260c1bcd559be3a81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1f1b64cc24e8a528f06e088d59beabc
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([256, 512], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0592b37a70f75767330d9adb8cebcdb0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1024, 2048], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8f8f86601e71e0ec5b17dde726f410e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0592b37a70f75767330d9adb8cebcdb0
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1024, 2048], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_880420458e8f0925ee799b9d176629c5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 192, 320, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1454067d86023a493801d2d44cf52deb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_880420458e8f0925ee799b9d176629c5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 180, 320], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_dce3a74c154ec29d22075ed89a7151f8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c84186d18b97086352d6374bca7a7196(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dce3a74c154ec29d22075ed89a7151f8
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 48, 80], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_46eeb60c7157a6578a085fdde56e39d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dce3a74c154ec29d22075ed89a7151f8
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 96, 160], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2ca05cd521f1747e5679a5dffeb08c8e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 180, 320, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a1660bf94874f19e5faf782e0a2027a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ca05cd521f1747e5679a5dffeb08c8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 192, 320], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_56a696964211f1dbbae976065755d726(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [float('4'), float('4')], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9e8d9770ebb82a7ba27bd86eb0b58fb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_56a696964211f1dbbae976065755d726
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 180, 320], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_034319fb0566f3c1a9ed128afefc1094(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([32, 16], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d83e9c837df4ea435401665b3bcdf67c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_034319fb0566f3c1a9ed128afefc1094
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 16], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d7eaeb612f6d82040089cff5a10725cc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([32, 16], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 2, 2], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a14bb2d2c288a88861139a0c3b6916d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7eaeb612f6d82040089cff5a10725cc
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 2, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 16], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_03caefdcf590715b999712f5932cac55(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([32, 16], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 4, 4], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9cc7e32649196e2253995c3eb932a8ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03caefdcf590715b999712f5932cac55
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 4, 4], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 16], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_48bd3e96ab59deaffa7294293e93be05(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([32, 16], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6550b234715d5ce468618894125ea191(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48bd3e96ab59deaffa7294293e93be05
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 16], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 16], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_41e85ba5235ebee46e4aada1f4574cd1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([64, 32], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_58c2788a3ad3b0bdc69cebed4cce00e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41e85ba5235ebee46e4aada1f4574cd1
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 16], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([64, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6996e84b02d8894f49298efee8b730e0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([128, 64], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ffb0acb1aebcd196b58f7570f5307980(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6996e84b02d8894f49298efee8b730e0
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 64, 32], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 64], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_12cbc013082d895cae1f977a57c074e3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1024, 512], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 19, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_35c5f53b3e30ccba0e8577c462209a80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_12cbc013082d895cae1f977a57c074e3
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 128, 64], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1024, 512], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a1bb88e839fb4930a7a583e87e5c2a6c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 8, 6, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 160, 4, 3], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_266d41b882a408893a40e631dd980ff1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a1bb88e839fb4930a7a583e87e5c2a6c
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 4, 3], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f535319f8482cae4171bd38d0011be86(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 16, 12, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 80, 8, 6], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_615b988cb61710adf527769b0502a284(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f535319f8482cae4171bd38d0011be86
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 8, 6], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a4b39f595dadab95bb4692aabfe6edab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 32, 24, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 40, 16, 12], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6271e21ae827b2ec7fdf7f546b616685(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4b39f595dadab95bb4692aabfe6edab
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 16, 12], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_40f7aa697075b848e060dfa93c93ee91(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([128, 128], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 32, 32], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d8a065bc8f08dc8a104618b736e4c1f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40f7aa697075b848e060dfa93c93ee91
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 64], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_613374f1a84bf219cc01c138562db57e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([128, 128], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 16, 16], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_601ef48d76a6ad7e99031a541a73a18c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_613374f1a84bf219cc01c138562db57e
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 32], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f78088bfcb867e4059a74640bcb7d5d9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([32, 32], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 8, 8], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9b4a0d9e6638bbc3d073eaa39b38b585(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f78088bfcb867e4059a74640bcb7d5d9
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 16, 16], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_60afa9ae4f4ec7e558afd0cbe38014e7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([32, 32], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 4, 4], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6ddc5d4394319eae4c8aca14fbaae5c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60afa9ae4f4ec7e558afd0cbe38014e7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 8, 8], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_de35f706384766a83417ec1b4123811a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([32, 32], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 2, 2], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_66a00a064f3e95e98ab21ed9a7de036d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de35f706384766a83417ec1b4123811a
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 4, 4], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e43cb99efa5d57400f37c85ff61b8e75(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([32, 32], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1eeaaabf0eca591ebbeb3d1f0be84cb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e43cb99efa5d57400f37c85ff61b8e75
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e9e9b27e9f74f2063b4fa6bab836a87b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1024, 1024], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 19, 64, 64], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9061fb6b674e736daf10d9071de78f04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9e9b27e9f74f2063b4fa6bab836a87b
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 128, 128], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1024, 1024], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_061fda9b75a7cae19f2c1cdf7bfae7a7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([128, 256], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 18, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_84cd747bd8f8cc4afb18db6b0ad65b4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_061fda9b75a7cae19f2c1cdf7bfae7a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 64, 128], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b75991b364315c3318698ed34550f90d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_061fda9b75a7cae19f2c1cdf7bfae7a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 32, 64], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_66efc958dc79961d4ff33e7b4b9222f8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([64, 128], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 36, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9406e6e7079399bb44ff3e4bacbc9ba2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66efc958dc79961d4ff33e7b4b9222f8
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 32, 64], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([64, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a148dab215493432778017f1817922c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_061fda9b75a7cae19f2c1cdf7bfae7a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 16, 32], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2323e198cb6265d0e01b0b5a38b383ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66efc958dc79961d4ff33e7b4b9222f8
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 16, 32], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([64, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5d2b3ac436780b54fc51408a5b9a3b83(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([32, 64], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 72, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_79fe5c8e6f8803eb01c05885154fa3b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d2b3ac436780b54fc51408a5b9a3b83
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 16, 32], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 64], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d2dcdc61c725f237369d9a7e4960c671(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([128, 256], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 36, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e52cba7e06940978b67c96ace4f408d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2dcdc61c725f237369d9a7e4960c671
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 64, 128], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_3a508ae3c30bdb1a41c5bb7c063f7ff2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([128, 256], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 72, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e9356e4f18fa87178d53cbdb179deb10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a508ae3c30bdb1a41c5bb7c063f7ff2
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 32, 64], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_45a073cea010b53e39514e2170cfd005(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([128, 256], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_98ac33dc7eafe0bb8af19c8082d917f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_45a073cea010b53e39514e2170cfd005
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 16, 32], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f137bf430472c7e45a9465a4cad16077(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([512, 1024], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 19, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a1a18120c8259ce6fcfad77f99824e63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f137bf430472c7e45a9465a4cad16077
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([512, 1024], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6e388e7a8d22f741669b7fefbef0cf99(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([112, 199], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0a14828e4b65b1464ee053787d85624d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6e388e7a8d22f741669b7fefbef0cf99
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 14, 25], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([112, 199], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_71844b2ee41bb377fe3b9ad44137612f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([224, 398], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b8bb967a9a7236347f5e41ecc1fe2976(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_71844b2ee41bb377fe3b9ad44137612f
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 112, 199], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([224, 398], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_892db6a56ed24399e53e52eeb47c57f2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([128, 128], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b5d8dad0dc6ff574ae5268caea2ebc17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_892db6a56ed24399e53e52eeb47c57f2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_03241e28cc68bc79c6bfe03eaed4c48b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1024, 1024], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 19, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6dcaa52baefbff63785304487a1edf17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03241e28cc68bc79c6bfe03eaed4c48b
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1024, 1024], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d74a64fa90bbbb8358e70138ead02e20(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([128, 128], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cc0f836634ae4800fe59fc297ed31d8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d74a64fa90bbbb8358e70138ead02e20
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_da674953ab1f2ba278bddbb96ee5f948(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([128, 128], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 2, 2], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ae2545f0242786b4fd7db00e0ece12de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da674953ab1f2ba278bddbb96ee5f948
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 2, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a91d88c8f391b3e13193d9bc0bc2327f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([128, 128], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 3, 3], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c1c58c08bfc8642b3b298407126336a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a91d88c8f391b3e13193d9bc0bc2327f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 3, 3], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0e32b5c3191a3a5403eaa85aec612aac(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([128, 128], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 6, 6], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8b4d4b8b0ca15ce5f24ec4714633ced4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e32b5c3191a3a5403eaa85aec612aac
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 6, 6], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f0e51eefd9f3cd14b2c64490f734d469(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([128, 128], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_185d7c1c524913ec6d71b507846e380a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0e51eefd9f3cd14b2c64490f734d469
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 128], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ca933ef637ea434a4d21f50401ea1222(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([256, 256], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_83909abc97f36f3b234e3ccdbab0d0c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca933ef637ea434a4d21f50401ea1222
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 128], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([256, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6dcc3d76c233ed2e460ba9c00d702067(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([256, 256], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ce45941fd13b3ad81842d77e1630e7de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6dcc3d76c233ed2e460ba9c00d702067
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 128], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([256, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_39dfe5e0e6a028a8ef3ce45517f7d680(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1024, 1024], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 19, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0ef00f4586e66c6b86c5f7d62fe6b8b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39dfe5e0e6a028a8ef3ce45517f7d680
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 256, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1024, 1024], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ef1eb0125155fc05ad495e520536d86d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([128, 128], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b4d0f617b2092e6d8a390480eee3d7ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef1eb0125155fc05ad495e520536d86d
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4383c6749837824b84a12412f41c94c6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([128, 128], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 2, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3894983d4c1bd71ab23afa24a4f933e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4383c6749837824b84a12412f41c94c6
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 2, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c58368e313af0836c07f6ffb9cb5a8a3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([128, 128], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6a366a648e7cb01e63b2377bbd517402(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c58368e313af0836c07f6ffb9cb5a8a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_63298bdf8fbe93e33e8b3e26d84a26a7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([128, 128], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 6, 6], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_330566a2df281ccf50fc6c5dc8083cf7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63298bdf8fbe93e33e8b3e26d84a26a7
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 6, 6], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4a021b2507a1ab2a538169b3a4bf39ca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([128, 128], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_50c3a3e92458465539cea4056713e747(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a021b2507a1ab2a538169b3a4bf39ca
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_132709e3eabd5abe367f8b861b614b20(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([256, 256], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_84f59971480d707bbb7161b12e81eac9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_132709e3eabd5abe367f8b861b614b20
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([256, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_fea722281f2dce644d44ba5eb063a90e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([256, 256], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3250c17905b3b78c22f95c99d1d139cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fea722281f2dce644d44ba5eb063a90e
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([256, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_20489a39e02c944bff21e3294cd8b349(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03241e28cc68bc79c6bfe03eaed4c48b
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 256, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1024, 1024], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_af414ce5c6af60b4ab21e61488338e29(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([128, 128], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b41841bc028ca170b34a442958b3d2e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af414ce5c6af60b4ab21e61488338e29
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 32], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b784851ef1bd96f1d875a390ee31fd82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39dfe5e0e6a028a8ef3ce45517f7d680
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 128, 128], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1024, 1024], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_aed562d04e5dc079e7bfb5dc6a89d8cc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([128, 256], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b6287ef3dff6693a8e5f507f273815bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aed562d04e5dc079e7bfb5dc6a89d8cc
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ed46516d811b6bd8a84c124671165256(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([256, 512], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5b22d4109a3eab03b26e1e500514afec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ed46516d811b6bd8a84c124671165256
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([256, 512], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_940eaf298719afab2c5bda303f7f220b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_13c99d3ebc90c7cdbceb0a695ecc0dd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_940eaf298719afab2c5bda303f7f220b
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a2194e69f11cdb63ee35c40b41fab126(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_940eaf298719afab2c5bda303f7f220b
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 256, 512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_624a7e4d9d739e962ce6567688fe392d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([256, 512], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_69ada097364bd937084623692632c72d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_624a7e4d9d739e962ce6567688fe392d
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 512, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([256, 512], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cfcd75310662b703a422c5f2709c0869(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_624a7e4d9d739e962ce6567688fe392d
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([256, 512], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_190dc77264362baa34b2dbbe8ffa57f2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1024, 2048], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 19, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_87171788cb5c8366ccfd8628dc66f680(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_190dc77264362baa34b2dbbe8ffa57f2
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1024, 2048], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2f837498191dc5c2e04751cc393e0719(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([64, 64], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4bca9bdac40aad8fd2adf1b6bdb0ceee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f837498191dc5c2e04751cc393e0719
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([64, 64], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_180c0d69f3cb0ddbf257d5901d91191f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([128, 128], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_53b983be16e0c951f8827bceeb38cb38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_180c0d69f3cb0ddbf257d5901d91191f
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f930254b52fe24b718034416414fec50(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([512, 512], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_43c56de501cc5ea6e40fe7d4bbf20a33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f930254b52fe24b718034416414fec50
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([512, 512], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_03d61c385342339a6ffda1d438b61b6d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([112, 199], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cc854da1c95c73f50dbe11105ce8229a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_03d61c385342339a6ffda1d438b61b6d
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 14, 25], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([112, 199], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_62ae168b04fc30e481de70e9bee82a55(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([224, 398], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5cf2fc96b5104dcb32641f0f137cb6f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62ae168b04fc30e481de70e9bee82a55
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 112, 199], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([224, 398], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5801d24d88bca06d773c68c56d9b4f6b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 32, 64, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 64, 256], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_20cd196fdcf766c83e1bd7745f6147cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5801d24d88bca06d773c68c56d9b4f6b
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 64, 256], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_914aefdb3914ea6edfe6429acb1d1fcc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([32, 32], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8b4a114a99a492d3cab949e236766835(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_914aefdb3914ea6edfe6429acb1d1fcc
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_35e331887e082dd3130b3291a44dd4c8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([32, 32], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 2, 2], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0947769538b946abdd360daaff8ff60a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35e331887e082dd3130b3291a44dd4c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 2, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0aceaa02f90223eb2c8372946aeed418(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([32, 32], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 3, 3], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f100419b3d5c9bc524fa2a1bd212bbdc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0aceaa02f90223eb2c8372946aeed418
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 3, 3], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ba65409e47a0da2939a4bffa8da85eec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([32, 32], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 6, 6], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_be1c2aa94aa80bb600351c9486fa54d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba65409e47a0da2939a4bffa8da85eec
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 6, 6], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a60cf2dc043743d069f00f594535df02(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([64, 64], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a6820f0807da35fbeb16934e9be9ac6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a60cf2dc043743d069f00f594535df02
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([64, 64], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b7f6294552f781e8712c8d4391c14c59(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([128, 128], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9d21391e838cdfda823eb54c5d9916d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7f6294552f781e8712c8d4391c14c59
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 64, 64], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_53e8e1d0ac4baabecba81e0e7a159af0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([512, 512], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9633ad82d26a2be5f2a7f3fdc2d5248d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53e8e1d0ac4baabecba81e0e7a159af0
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 128, 128], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([512, 512], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8ed0a785e703baea543f177cbd2f8f1f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([128, 256], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 18, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1d5c0b5101a1171e226f29949a62da4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ed0a785e703baea543f177cbd2f8f1f
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_16ca5746e4439ed7985e798c3cf5da9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ed0a785e703baea543f177cbd2f8f1f
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_fbe4836adf88d5129481c9b8971d99f5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([64, 128], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 36, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f9dcf2db5cb725d6785d2d3f9c8d91ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fbe4836adf88d5129481c9b8971d99f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([64, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_54559df50a1bd5610ed7a1a2b1772675(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ed0a785e703baea543f177cbd2f8f1f
    def get_inputs(self):
        return [
            paddle.uniform([1, 18, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5447dcf39176d0732ab5cb2918355220(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fbe4836adf88d5129481c9b8971d99f5
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([64, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_3df4308217e9cfa9c465fa8cb987bde7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([32, 64], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 72, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_413e8b9604336f5cf0e015b045835470(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3df4308217e9cfa9c465fa8cb987bde7
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 64], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_3d046c23835fa9ff22f35acc2a7fc4b6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([128, 256], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 36, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6078af8f43d1ae79e9a0835861fd1e67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d046c23835fa9ff22f35acc2a7fc4b6
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 64, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1d6e184dd196aef273df64a190d1a047(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([128, 256], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 72, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_42240ac6e7a2cea7f0632b740324fee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d6e184dd196aef273df64a190d1a047
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 32, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_68ae81722d1dc1b061cb478828317440(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([128, 256], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d9342cd3251d8fb884a9b9ff8f85d4b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_68ae81722d1dc1b061cb478828317440
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 16, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_37f78eddaab0509ecec97e93f9edc410(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([512, 1024], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 19, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0db3f857ae62431b47f45984c091ff60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37f78eddaab0509ecec97e93f9edc410
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 128, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([512, 1024], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_346dc76e7022c40c5a7f5338c4e89f6e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 8, 6, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 160, 4, 3], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b31be8aa0d84c9db20cbd1018ecbab04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_346dc76e7022c40c5a7f5338c4e89f6e
    def get_inputs(self):
        return [
            paddle.uniform([1, 160, 4, 3], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_bd9d3244333de7ef0e3920653ed6007b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 16, 12, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 80, 8, 6], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_57052eacd590b8725b365ea02afaec8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd9d3244333de7ef0e3920653ed6007b
    def get_inputs(self):
        return [
            paddle.uniform([1, 80, 8, 6], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_48abf86608ab8fd99acb8bddb434c191(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 32, 24, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 40, 16, 12], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_daceab60b5da579f339e8162c88fc098(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_48abf86608ab8fd99acb8bddb434c191
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 16, 12], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_abaa0b20cb269aaeda7e155bfd3a177f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 32, 100, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 2, 6], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5867163b8e49e61b9eb9f02a1e986fc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_abaa0b20cb269aaeda7e155bfd3a177f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]]], dtype='float16').reshape([1, 1, 2, 6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e9d92523ba055a25067d83f3dc47ad20(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 32, 100, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 2, 6], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_84bdc2b60fe822c8a6db7c2abe25770c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9d92523ba055a25067d83f3dc47ad20
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]]], dtype='float32').reshape([1, 1, 2, 6]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2495fa3bbe4d553d1323a77173db7cbb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([32, 16], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_00f896e6678b8df75f38a23915e460e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2495fa3bbe4d553d1323a77173db7cbb
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 16], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_59615796d14a01fef11bb3c9906da34d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([32, 16], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 2, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_78a1b08bd723c2a55a030bab18414bee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59615796d14a01fef11bb3c9906da34d
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 2, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 16], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1b31f209f1a144a15d63c7ddd8dd4f3f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([32, 16], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 4, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_326c2abe02983f2130cdc59a400efd0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b31f209f1a144a15d63c7ddd8dd4f3f
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 16], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c7eb3f79db5acda0763836470c668409(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([32, 16], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6916f2cc1caaa9ccb880c7cd92736954(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c7eb3f79db5acda0763836470c668409
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 16], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_be791f9ea57b63973f807f9c131c73a3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([64, 32], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2d13aefd0e31b7e63e5bde8ff2b9ae0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be791f9ea57b63973f807f9c131c73a3
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([64, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ff3c6f15717c7bef9ec0f4a61fe49be3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([128, 64], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_403f08c1b067de62901e31e3ce1e549d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff3c6f15717c7bef9ec0f4a61fe49be3
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 64, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 64], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9b3c2c84559d4eee91eb629f385e7858(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1024, 512], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 19, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7a9affed1fc2ad54d5c416298354d44b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b3c2c84559d4eee91eb629f385e7858
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 128, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1024, 512], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_de856ea2ffd60a625dd7a831014695b9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([32, 32], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cd8172a438a3e2c9e329ecfe84e17900(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de856ea2ffd60a625dd7a831014695b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_045089399ab59978fd6d988924cd69fb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([32, 32], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 2, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b79ebda09a9542cc0c5539dbcbfc2693(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_045089399ab59978fd6d988924cd69fb
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 2, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e3c9b8da903978a9a92684bf132441e5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([32, 32], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_38791f19e09caa4f95da497bb76a6036(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3c9b8da903978a9a92684bf132441e5
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_cbb306699059893a3ec8143a4009b288(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([32, 32], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 6, 6], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6fe3e58de43e4f7f30c338003d73918a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbb306699059893a3ec8143a4009b288
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 6, 6], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c4d6b5feb3b141df5ecac2262f55007c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 32, 64, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 64, 256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7b20d2b21e0f2145fd43e841aba097a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4d6b5feb3b141df5ecac2262f55007c
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 64, 256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_7445f5c34d673cbc8710e6f763da1b41(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 192, 320, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 180, 320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3b896b0ec63df107c31d3f953993ebf1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7445f5c34d673cbc8710e6f763da1b41
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 180, 320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_032b18a2febf9b76a94d03ebed594a2b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 48, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a4b55bafcfd9f1aca0756d1ecc3ad86f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_032b18a2febf9b76a94d03ebed594a2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 48, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_7bc9145b7dcec06556f4089112db9d96(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 96, 160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_99686597650ce4a7472d9e2f8f9df4bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7bc9145b7dcec06556f4089112db9d96
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 96, 160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d73acdaf2ea9a62b3f6ebeca7b07c36e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 180, 320, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 192, 320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dbd47e6477fe6c90ab74ad8a6194b4e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d73acdaf2ea9a62b3f6ebeca7b07c36e
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 192, 320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c194639392c95be3172d2321f7b5187e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [float('4'), float('4')], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 180, 320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_61853436382819bf2ef387752ec627d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c194639392c95be3172d2321f7b5187e
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 180, 320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9173798637cd97fdda22380d60812b40(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([128, 128], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 32, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4ba260da49c9f7b4f720068999734acb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9173798637cd97fdda22380d60812b40
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 64, 64], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_212056eff99d22b10eaf74ac6b502b18(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([128, 128], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9ce22bf9e71e64e912e6f5c916104d40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_212056eff99d22b10eaf74ac6b502b18
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 32, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5bf0bfc013cd787d348535fcf316a969(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([32, 32], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 8, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e1273d5ab2f19b0e70297ed8625fe126(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5bf0bfc013cd787d348535fcf316a969
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f7dac40ee9bd03d47e14cde22a3b92df(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([32, 32], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 4, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f4dc6fc574ad55c069fa95d30548223f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f7dac40ee9bd03d47e14cde22a3b92df
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 8, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8bc0e92b19fc5b3248c993c4f5269292(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([32, 32], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 2, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_abba1921a2d303aeed905431bb15d14b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8bc0e92b19fc5b3248c993c4f5269292
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 4, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e300224e87f4d8cf3df571ee9d37decd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([32, 32], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 128, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2db26cd744a153a623107d2f175f5b5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e300224e87f4d8cf3df571ee9d37decd
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([32, 32], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2c47c530cd8c97e5f55de284b8bfc604(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1024, 1024], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 19, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c797df86c4d9f460ab25e49fd36ba432(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c47c530cd8c97e5f55de284b8bfc604
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1024, 1024], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a099a416dad6a3a811e1de01cb958891(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([128, 256], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f96403ed4a8ef11dde9b7b3e6cf69f8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a099a416dad6a3a811e1de01cb958891
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([128, 256], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_975621ac082c18105ad5df7bd0e33efb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([256, 512], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_aec07d6e01e2242df62040953e2d8b30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_975621ac082c18105ad5df7bd0e33efb
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([256, 512], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a63e9735040763b14b9d27ada4d392ff(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8475248081b43ac8533f1b2a7ae25fca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a63e9735040763b14b9d27ada4d392ff
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 128, 256], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4c03a12a0674e3c88c4dc8261c75c795(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a63e9735040763b14b9d27ada4d392ff
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 256, 512], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_3a306ca81fd306410a80f679aba7eb56(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([256, 512], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e43011715ea5f797e8d41e082c293f35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a306ca81fd306410a80f679aba7eb56
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 512, 1024], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([256, 512], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dcbe59982c83173cb3950766a18c6f94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a306ca81fd306410a80f679aba7eb56
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([256, 512], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8464864985fcf62cc5e4c324e2373f40(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1024, 2048], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 19, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a75303bf946be229b1875187337de947(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8464864985fcf62cc5e4c324e2373f40
    def get_inputs(self):
        return [
            paddle.uniform([1, 19, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1024, 2048], dtype='int32').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a44f7cb9a01555e745286543a7a785bf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 192, 320, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 180, 320], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5f569539692efe23203882ac5132ef82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a44f7cb9a01555e745286543a7a785bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 180, 320], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_38e2fcafc5bdd980968d0f6ffaaed50c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 48, 80], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f15cc669da85cfac1e7c62318ab4ea5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38e2fcafc5bdd980968d0f6ffaaed50c
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 48, 80], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_faf7f3f92b62101b88f06763fb5b0fa1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'bilinear', True, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 96, 160], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_54c1b283949c99e0af6dad39c2da72c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_faf7f3f92b62101b88f06763fb5b0fa1
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 96, 160], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6ff1d290e44a5e5653eb9f7a3e0a9908(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, 180, 320, [], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 192, 320], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8fa449684eaabdf52cf54e97303872da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6ff1d290e44a5e5653eb9f7a3e0a9908
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 192, 320], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a24272f1a6b069055e624c83f3b4370f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        input_1 = None
        input_2 = None
        input_3 = None
        return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [float('4'), float('4')], 'bilinear', False, 0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 3, 180, 320], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_420f64c523d1130b2484a06eabe9b104(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a24272f1a6b069055e624c83f3b4370f
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 180, 320], dtype='float16', min=0, max=0.5),
        ]


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