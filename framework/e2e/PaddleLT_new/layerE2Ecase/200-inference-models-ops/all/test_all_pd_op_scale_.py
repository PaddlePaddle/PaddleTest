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
class PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_88dc0bad6ba7b34623e3acd8244de75a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.054565705358982086], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_62ac7d4b9139c1ec18e94d31dcfa59ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 256, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2397179752588272], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f78927cfd9c0901b958968f0a3a58d9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([196, 4, 16, 16], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4900263249874115], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_14202fc2442153f7432c1efe06678284(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 196, 49], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.15890726447105408], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0977bdac2b848a6b9560943e7ea74792(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_96d96474eec7a7209cd0fc0cb2a942b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0977bdac2b848a6b9560943e7ea74792
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 80, 80, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.27328571677207947], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_cc9406d6678bff2b2dbc460c1015bbfe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('-0.5'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_12e8c1984bc892a9049356a045e6740a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc9406d6678bff2b2dbc460c1015bbfe
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ce2dfbbd8cf644e56dd9529169710001(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4588a9285f41836a655bc6346d7f23f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce2dfbbd8cf644e56dd9529169710001
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 40, 40, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.46604785323143005], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_02aff8bb67b40b635be3683718da3316(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([56, 2, 56, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2251664251089096], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_45fdbb5a58f9fbcf117174854eded3a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 26, 26], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2397179752588272], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ec4e957aa7db475c3c4161b2190f7a9c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 196, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3837188482284546], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_96138fe3805b48a39c61f718145ed32e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc9406d6678bff2b2dbc460c1015bbfe
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3d14b3734914b5c7c8656c7e60432ee8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([4, 16, 144, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.27207040786743164], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f527ad8ab7a192f5752e4c0c71ebf21d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5d1ec12a02cb9a620dcf6bfd619dd3c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f527ad8ab7a192f5752e4c0c71ebf21d
    def get_inputs(self):
        return [
            paddle.uniform([1, 25, 25], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.34655219316482544], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a38cf865df5cfd3b84f4bae48ed2ab7d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('1'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4bdbb1d6748681a6bd211800e96c45b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a38cf865df5cfd3b84f4bae48ed2ab7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e362d7e6fdcd7eefde5c0e473286fdb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.08247071504592896], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_46d7a2f5d55174c3dd72d73d977e86f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a38cf865df5cfd3b84f4bae48ed2ab7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c2a0c246f26157857be7fa6040734d08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.054565705358982086], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8fb5b7a05966ae7202f9071e98942ed0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([2, 8, 98, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17822237312793732], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2990b9b8a9ed911f1d78a4b1eba8cddb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4858695864677429], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e705521bfe1f11e10326662be8cd288e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 197, 197], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2784574031829834], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bc3aede4f53313adb6a739d6bd7e6b3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_58472b74fc21ff9fd9396aa2362ec855(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 197, 197], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3810800313949585], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_96f655515e273fb04202aa45664a6b41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([2, 8, 98, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2251664251089096], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f0642cee295a05479288b8267cd1ed14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 196, 196], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.039581477642059326], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e9c3f07e8238744122b167ace3b98783(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 180, 320], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.42274439334869385], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dd5f6c64d264959b123ba181a1c9e79c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([14, 4, 56, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17822237312793732], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2f265e95a70bb268fbda1f9b746cd173(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('1'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3dcdfbf4e969f1516090ebbd9c129eba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f265e95a70bb268fbda1f9b746cd173
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_50c3b595f8baa5c82803e277f0c40d6c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f527ad8ab7a192f5752e4c0c71ebf21d
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.21983124315738678], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e44eacf24c8523b82aa76145aaf57c08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 196, 49], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.41277825832366943], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_774e43ee0978252bc12bb87b2f098722(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 49, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.27348795533180237], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_403a5befecf3c66d90de560d4022919b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1fd6d6185e6c9d27eff7a4a36710a7b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 49, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3837188482284546], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0dcf408ef8d17d1b88342ee924fa36d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 49, 49], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.15890726447105408], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d8f67b4e979e5bda289f780610443ae1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 192, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4670775532722473], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_128bea1dd0b2d9b0b5bc4933d4959391(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([4, 16, 144, 32], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.14657902717590332], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1810e6c8ad42b9555104ed0ace60aad9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f527ad8ab7a192f5752e4c0c71ebf21d
    def get_inputs(self):
        return [
            paddle.uniform([1, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.49733033776283264], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_90a6611891e0812d19bf9c835726d6ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9d51f8a52d6686be09922b630c575fe0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_239c28715519ce277bd6ee6925910e0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d51f8a52d6686be09922b630c575fe0
    def get_inputs(self):
        return [
            paddle.uniform([8, 256, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4651494026184082], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_36471c1ada27275ca664798bfcb1babb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc9406d6678bff2b2dbc460c1015bbfe
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_36666a19e172ec510f4ad3ea858b6f97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 196, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.016803564503788948], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_72609f6e19c724b8bcc6c533fce3e461(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1769556850194931], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bcd7efe17ab2c6f1304e5b3c7eeec275(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 16, 60], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4147603213787079], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0fb48f3484011138c921d43da0355f58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fcac596e40f13526715e44e299220c21(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 49, 49], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.09803877770900726], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_97b6825aa98fdc2618b033f9f26bf1a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 577, 577], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.22600552439689636], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_903380696229d6ab1cce07f349543ea6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f527ad8ab7a192f5752e4c0c71ebf21d
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4322921633720398], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e6fb30649f12b174698d786b78a9d124(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b51cbbbb3acb0779b2ca5e6e10e052bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.054565705358982086], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5e1feb1e09c532fd9f89cf4a4c4c81d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.14158113300800323], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_15730a612e08de53e1161e7327eba804(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 3136, 49], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2799938917160034], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ff15b3b0acf6fd59d587f59e9abf15d5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('-1'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f87c699deabb45e5f76a5ffdbdfb861e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff15b3b0acf6fd59d587f59e9abf15d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 160], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.42274439334869385], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_159449eeaf474a80257017ecac5c7b55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d51f8a52d6686be09922b630c575fe0
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 30], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.18577031791210175], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1bdc2652643f94045d0f3f6f3e9efa4c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('-0.5'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8aa46bdaa4abbba7e6e660ef36f4ee79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bdc2652643f94045d0f3f6f3e9efa4c
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fedbffc599952d17e50016feedc2e0a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 196, 196], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.09803877770900726], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1dcf637c3da4db15f0866f01487118d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f527ad8ab7a192f5752e4c0c71ebf21d
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.15963655710220337], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_91cd37171454962c2cec2615d3420b7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f527ad8ab7a192f5752e4c0c71ebf21d
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3119860291481018], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c02875565038e74917daa167a68a7735(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 197, 197], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.35620564222335815], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7e8ae9bc93b558e53ae3c56dbe12c957(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc9406d6678bff2b2dbc460c1015bbfe
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5b75a07f5c997c1e6aede2f2349f35df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f527ad8ab7a192f5752e4c0c71ebf21d
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4670775532722473], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7fddca5fca8775ce5cf3c6c08cb9a94c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 197, 197], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.44839635491371155], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_22ef50dbdcb479becc854c90690bef8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e001cb7b75b37e5d3e14f80f5ebfa5fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f527ad8ab7a192f5752e4c0c71ebf21d
    def get_inputs(self):
        return [
            paddle.uniform([1, 25, 25], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4551265835762024], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_53d58fd5ff6b85035a89a19e541bb92a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ff270172b6bc0dea5b67dd597bf15d65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53d58fd5ff6b85035a89a19e541bb92a
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0.016806144267320633], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1c4eeba18729982fff5a6b0f30a82ae9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f265e95a70bb268fbda1f9b746cd173
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_aa3c8117bc30886c37b79f0f0a328e3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.14158113300800323], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2d917dfb01c5c835b11f56320a636c01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_aff2cb8292404abf8b5af6bddb65de41(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('1e-06'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float64'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e5e4ba90198b471014e2782fcff6fa12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aff2cb8292404abf8b5af6bddb65de41
    def get_inputs(self):
        return [
            paddle.uniform([3200, 20], dtype='float64', min=0, max=0.5),
            paddle.to_tensor([0.0033531691879034042], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f4503ca346776cd623b502122c1e8379(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4858695864677429], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6e803c7126f11720c709c65cba71e8d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 196, 196], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3803968131542206], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1ae88ff88349db134ec247f3ec6e117a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a38cf865df5cfd3b84f4bae48ed2ab7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_72bcd65016dbbb849765122d55ac3995(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f527ad8ab7a192f5752e4c0c71ebf21d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.21536855399608612], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f485cbcd48ee83337a55ed76d5a643ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d51f8a52d6686be09922b630c575fe0
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 160], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2011336237192154], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_47723b80f7d6ca83660c04858dde9ae2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bdc2652643f94045d0f3f6f3e9efa4c
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ddc8ae767cdabb983a7e06aeeae5466e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce2dfbbd8cf644e56dd9529169710001
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 80, 80, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3051380217075348], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1aa6af613065edb17e05380c4ad3e6dc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('-1'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bd1dc28f82d5085d0e0ebd0a1b8279c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1aa6af613065edb17e05380c4ad3e6dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1769556850194931], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8c001ff15f1251093e87c7c6cdb02087(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_883cf795005703a01384adaecd29f20c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c001ff15f1251093e87c7c6cdb02087
    def get_inputs(self):
        return [
            paddle.uniform([1, 38], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2756720781326294], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3cba7a3a7e2eb0396f1b853436488d25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f265e95a70bb268fbda1f9b746cd173
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.0481078065931797]], [[-0.03962375596165657]], [[-0.041207537055015564]], [[-0.041738662868738174]], [[-0.03966488689184189]], [[-0.043501757085323334]], [[-0.04142988473176956]], [[-0.04110235348343849]], [[-0.040328122675418854]], [[-0.04362419620156288]], [[-0.04235588759183884]], [[-0.04771324619650841]]]], dtype='float32').reshape([1, 12, 1, 1]),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f46642688516c70d513c1bbf929b6f73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d51f8a52d6686be09922b630c575fe0
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 80], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2011336237192154], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_99901d2c4be35058a06b71b1e4ac029e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 3136, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.016803564503788948], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a1fd0db18d169d757bdafd0b7a7070c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d51f8a52d6686be09922b630c575fe0
    def get_inputs(self):
        return [
            paddle.uniform([1, 25, 38], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2756720781326294], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a4c7f7259b8e29dfebff60368e3591a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce2dfbbd8cf644e56dd9529169710001
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 3, 49, 49], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.41277825832366943], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_acc2dc1289ed504e66283a6f659d1466(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 49, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.446737140417099], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dcf649f8bbdc2dbc08b40db8ff324370(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1b7bcd8eefb6420fe97f0fb6997d4a9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 197, 197], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4892241060733795], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f4672bc77052e8cc2f723f7f3b36db00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a38cf865df5cfd3b84f4bae48ed2ab7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fdfac1ea4ec2f5c83e51696edb800dbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d51f8a52d6686be09922b630c575fe0
    def get_inputs(self):
        return [
            paddle.uniform([1, 25, 25], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2309761345386505], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_81ad6582036868893ece52556d4927f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 784, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2404094934463501], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_65c486362270dbc119efd275f1e04160(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a38cf865df5cfd3b84f4bae48ed2ab7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4b69f5160e894e19b843ab1628a0925c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aff2cb8292404abf8b5af6bddb65de41
    def get_inputs(self):
        return [
            paddle.uniform([3200, 20], dtype='float64', min=0, max=0.5),
            paddle.to_tensor([0.22651496529579163], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9ef1f8476d7603734724f18e0605a251(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 49, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.039581477642059326], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bcd567396fc6d5321a14532b1421733e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bdc2652643f94045d0f3f6f3e9efa4c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.6282869577407837]], [[0.7110034227371216]], [[0.8070393204689026]], [[0.7731010317802429]], [[0.7212677001953125]], [[0.6343787908554077]], [[0.8950406312942505]], [[0.7803258299827576]], [[0.688999354839325]], [[0.7249688506126404]], [[0.9861186146736145]], [[0.857657253742218]], [[0.720317542552948]], [[0.8511000275611877]], [[0.8740426898002625]], [[0.8391191363334656]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b1a6db2e0996f94404952db4d85e7679(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d51f8a52d6686be09922b630c575fe0
    def get_inputs(self):
        return [
            paddle.uniform([1, 25, 25], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.36853480339050293], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_893a746b7e0de28fa235bd34336c0e8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f265e95a70bb268fbda1f9b746cd173
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6eb36d7c60b58daa86016c29991e4be2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 784, 49], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.49470216035842896], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7e9efc23ba66feed49eb9c5c73f779c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 3136, 49], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.15890726447105408], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d240d3e3b85987ba50784b5e100cf034(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d51f8a52d6686be09922b630c575fe0
    def get_inputs(self):
        return [
            paddle.uniform([1, 25, 25], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.002016119658946991], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1fecfbf9f8d004aa242d9989f36ef751(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('-0.5'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e59632443e9336e1a6dc35a46d5871e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fecfbf9f8d004aa242d9989f36ef751
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 80, 80, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.03263209015130997], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_333572e7ac65003d359cb2d7ebcb77b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 784, 49], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2799938917160034], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_47f4b49ff58df88be4952ff31a55ac17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.3299209475517273]], [[-0.30752936005592346]], [[-0.2815321683883667]], [[-0.2907193601131439]], [[-0.30475080013275146]], [[-0.32827186584472656]], [[-0.25770995020866394]], [[-0.2887635827064514]], [[-0.31348592042922974]], [[-0.30374887585639954]], [[-0.23305489122867584]], [[-0.2678297460079193]], [[-0.3050079941749573]], [[-0.26960480213165283]], [[-0.26339417695999146]], [[-0.27284806966781616]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.to_tensor([0.14158113300800323], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bef683b2d12088c997bb4888f052c8db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 192, 320], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2011336237192154], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_142b0026edd5cae6412b9a8ff6d1cb79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 256, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.08247071504592896], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dff1ecc37d9f440a77d7e90609e90da5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d51f8a52d6686be09922b630c575fe0
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 160], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4706328213214874], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c83c043495036fbecab2bfd74dd05ca2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 64, 48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1475658416748047], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_509c25620d03b3f937731a84d43b6359(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 196, 196], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3135160803794861], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a77c94b191ebb776153b46a29d20e5f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 196, 196], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.27348795533180237], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8afcf582c7f157d824e71a402606fb41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc9406d6678bff2b2dbc460c1015bbfe
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.6533203125]], [[0.619140625]], [[0.76416015625]], [[0.72021484375]], [[0.7841796875]], [[0.6435546875]], [[0.71533203125]], [[0.67333984375]], [[0.6591796875]], [[0.70361328125]], [[0.6748046875]], [[0.775390625]], [[0.76318359375]], [[0.67919921875]], [[0.7216796875]], [[0.73095703125]]]], dtype='float16').reshape([1, 16, 1, 1]),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2b163ba3a7d6b0e511bc03b5917a21a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([196, 4, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.22978518903255463], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f08ccb3038a24cf1d9351c782189295b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d51f8a52d6686be09922b630c575fe0
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2540014684200287], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8f583b3417b3f811c39dc8a5dd3c25e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 197, 197], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.006860947236418724], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_de8323d7d9457e7d59ee907b072eb98c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 96, 160], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2011336237192154], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9500a4f7ff5f323efe9745cfc88ec0a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.14158113300800323], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bab982e2fdd1809a8f121e9999d41418(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([64, 4, 144, 32], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.14657902717590332], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cea33858bc7900de14b19ab11b6c5374(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_baca0cf6f731dc7aca43fdc7111303d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc9406d6678bff2b2dbc460c1015bbfe
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_04e30dc579e697d48e7979774742b16d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f527ad8ab7a192f5752e4c0c71ebf21d
    def get_inputs(self):
        return [
            paddle.uniform([1, 25, 25], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3368866443634033], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4104f62e9b53c5ade9fe3ca9d41ff175(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 196, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2404094934463501], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e7d352f5ea11e8353f4f3b9fb2c1ebc3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bdc2652643f94045d0f3f6f3e9efa4c
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_7eb9f9d9e03db4054dc2c88afd4ebee1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('1'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8dfd587bc6cd2bb82f3cec39fedae576(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7eb9f9d9e03db4054dc2c88afd4ebee1
    def get_inputs(self):
        return [
            paddle.uniform([1, 26, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3119323253631592], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_341aab8b8063f35687889f7424a868cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d51f8a52d6686be09922b630c575fe0
    def get_inputs(self):
        return [
            paddle.uniform([1, 26, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.38372936844825745], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_69995b93acbb1327da04086cb9f25a84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f265e95a70bb268fbda1f9b746cd173
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a7252af08b3a5ee765a8debe2b20531d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d51f8a52d6686be09922b630c575fe0
    def get_inputs(self):
        return [
            paddle.uniform([1, 2125, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4863912761211395], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8a870774be8641879f13c126a06879fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 196, 49], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2799938917160034], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_54e6294e230aa511f4644f30cacbaa52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f527ad8ab7a192f5752e4c0c71ebf21d
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 30], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3747236430644989], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b3258a277f82bacf21bb13faa65315fa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0caba653411aab2d6966e5a98628ce22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3258a277f82bacf21bb13faa65315fa
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 1024, 2048], dtype='int64'), 'int32'),
            paddle.to_tensor([0.19310657680034637], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_83568d21c75387803b7735d908e301d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.21432095766067505], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3da3b62d2f689337f687be3484f7aa37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce2dfbbd8cf644e56dd9529169710001
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 12, 49, 49], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.41277825832366943], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5a55893bf700668c79cdf40156696518(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0977bdac2b848a6b9560943e7ea74792
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 20, 20, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2297879308462143], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2ac9323436169764985f97cd47a289bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d51f8a52d6686be09922b630c575fe0
    def get_inputs(self):
        return [
            paddle.uniform([1, 26, 37], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.42372840642929077], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_202f3d332d3255ce60b384213abc592a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d51f8a52d6686be09922b630c575fe0
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 160], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3401828408241272], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2193e72e7e36720e19b04f2840807f47(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('1e-06'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f241da1c39c0fd64945262dd0f1c0d4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2193e72e7e36720e19b04f2840807f47
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.11356471478939056], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8dae1eedccbc2d34461a8034a7e772d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0977bdac2b848a6b9560943e7ea74792
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 40, 40, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2297879308462143], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_62d4f554588cc5ab37eed68813a6e1e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 350, 25], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.19558078050613403], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bd74a64c3de5cce9689d49cdb2af5644(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a38cf865df5cfd3b84f4bae48ed2ab7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_422a3f4b809be0f57b92849ad4679ab3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0977bdac2b848a6b9560943e7ea74792
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 12, 49, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2404094934463501], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b39176189dae0b2f304d9b5ba19d11a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 196, 49], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.49470216035842896], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_538f25a6e7457a7754daf5a24361bdfc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 49, 49], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2799938917160034], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4889a84b3717c565cc75cf3834e52f7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.32212451100349426], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_87a9722d0863459bf20177a60edbe24f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0977bdac2b848a6b9560943e7ea74792
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 24, 49, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2404094934463501], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0b8cfaf6b923f5ca7c33efdca4b20230(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([64, 4, 144, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.27207040786743164], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eeee0719607e76b56bb10050cadc2498(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f527ad8ab7a192f5752e4c0c71ebf21d
    def get_inputs(self):
        return [
            paddle.uniform([1, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.22651496529579163], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0cd56504c9489bc25ebfcc5f448e083a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 577, 577], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4514082372188568], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3585ba8c8e4b8711d65d36c728d1f701(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f527ad8ab7a192f5752e4c0c71ebf21d
    def get_inputs(self):
        return [
            paddle.uniform([8, 256, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.07418585568666458], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_974b028adcfc3677408a9842845ec4a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3258a277f82bacf21bb13faa65315fa
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 1024, 2048], dtype='int64'), 'int32'),
            paddle.to_tensor([0.4215003252029419], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e613beb8e1be12b5dd2929b9779dc96f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0977bdac2b848a6b9560943e7ea74792
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 6, 49, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2404094934463501], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6e5aa9112b0aa387a0c1ef21de000895(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d51f8a52d6686be09922b630c575fe0
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 320], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3930826485157013], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4be5717ac1c140ee7f35a2ed285fef48(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_505ae223bf807ac105f78aa5799d7c93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4be5717ac1c140ee7f35a2ed285fef48
    def get_inputs(self):
        return [
            paddle.uniform([1, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.266884982585907], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c73cd4c7269315ad975bd29af0494be6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_adaf12e78cbd04f2fc8709892dc56408(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([56, 2, 56, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17822237312793732], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_843b977f653e2cbdcef3f401fbbf8450(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f527ad8ab7a192f5752e4c0c71ebf21d
    def get_inputs(self):
        return [
            paddle.uniform([1, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.05577254667878151], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8bbf9a4340eb4e1b986cf8ca98cd2bda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 256, 36], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.21719443798065186], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_48b3dc14ca472b7f1ef4cb3d38717201(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([16, 8, 144, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.27207040786743164], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6d1c7495245199822f861bf0a6e73221(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bdc2652643f94045d0f3f6f3e9efa4c
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7d5cb7e72c1aca38e1262268d4cf5711(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 144, 32], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.14657902717590332], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b69e97a4c48eee51a2e9315042ca4a72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aff2cb8292404abf8b5af6bddb65de41
    def get_inputs(self):
        return [
            paddle.uniform([3200, 20], dtype='float64', min=0, max=0.5),
            paddle.to_tensor([0.2728694677352905], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_83e59a3f540373c461afa7ea1013fd29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.44748714566230774], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_172f98b9e1c86358c81236fec937695b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce2dfbbd8cf644e56dd9529169710001
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 40, 40, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4837779998779297], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_297bd6d291ad99f0fefc4a515b4c38b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d51f8a52d6686be09922b630c575fe0
    def get_inputs(self):
        return [
            paddle.uniform([1, 180, 320], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2011336237192154], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4e031892718e14684992c59f9da45d0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.14158113300800323], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a9ddba178558f16f5f7133311ccff7b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f527ad8ab7a192f5752e4c0c71ebf21d
    def get_inputs(self):
        return [
            paddle.uniform([1, 26, 37], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3323487341403961], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_112cc57415986f6c68804276e63f9848(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 49, 196], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3135160803794861], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_11aa077311878c2b687aecef6c04990b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fecfbf9f8d004aa242d9989f36ef751
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 20, 20, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.03263209015130997], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_fc55dbaa1864d9c4113eda50840ef9b4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('-1'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4fc9ea32cbe3607405cbe6a04356538d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc55dbaa1864d9c4113eda50840ef9b4
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2728694677352905], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f09b14fd6fc8351b3b2ac3c669dccf8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 16, 60], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2064722180366516], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_914edf975d1dd2ff88cde9eafe9024e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc9406d6678bff2b2dbc460c1015bbfe
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_45c7d4447f65ef97533077e2d1f16fa4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.011260986328125]], [[-0.0105133056640625]], [[-0.01250457763671875]], [[-0.0118865966796875]], [[-0.0083770751953125]], [[-0.010894775390625]], [[-0.0131988525390625]], [[-0.00815582275390625]], [[-0.00984954833984375]], [[-0.01152801513671875]], [[-0.01430511474609375]], [[-0.0120086669921875]]]], dtype='float16').reshape([1, 12, 1, 1]),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_49109bc3f847224fa3fe747a0024ebe4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 64, 48], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2749481797218323], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a265d41a7228d3983188ca33ec88fd9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 784, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.446737140417099], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7e56326a44e3f45df17000d4e68aa4e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 180, 320], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1966366469860077], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c43e8786ab182b052e80a48ea9698ad0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f527ad8ab7a192f5752e4c0c71ebf21d
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.15963655710220337], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3776ee214df1a8ee7d558cdad2110c98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1aa6af613065edb17e05380c4ad3e6dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1769556850194931], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ab529a14a29b49a8513cf41f7b64bfe5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f265e95a70bb268fbda1f9b746cd173
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_502992ebd6e556e813d9d3cb9fb4a524(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 3136, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3837188482284546], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_51adba4cda11d7d2a4e26378bad02b80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([14, 4, 56, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2251664251089096], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_14412432f6b2c8fa14c8849c0cb397e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d51f8a52d6686be09922b630c575fe0
    def get_inputs(self):
        return [
            paddle.uniform([1, 180, 320], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3930826485157013], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4280d5d72f041a1e810e10487b107aca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d51f8a52d6686be09922b630c575fe0
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 4], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.18577031791210175], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4c5cd7ada0d3f165cfd91526d0873956(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('1e-06'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5d5744ea4b4616b9ccc5919897695b9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4c5cd7ada0d3f165cfd91526d0873956
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1, 1024], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.35355493426322937], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_73165bea716458b77299d29772bb5a36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f527ad8ab7a192f5752e4c0c71ebf21d
    def get_inputs(self):
        return [
            paddle.uniform([1, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4428243041038513], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_de34bf5b301f8a6d5cacaa3ff7abec8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f527ad8ab7a192f5752e4c0c71ebf21d
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.25278934836387634], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_db0d4e43b780983f78146c2b8614615f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.054565705358982086], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_621ae3d193df038712f3e8f1d98d10b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f527ad8ab7a192f5752e4c0c71ebf21d
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.48962774872779846], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8a6c8316f4ef58c1f7c78817d4d79ab1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aff2cb8292404abf8b5af6bddb65de41
    def get_inputs(self):
        return [
            paddle.uniform([3200, 20], dtype='float64', min=0, max=0.5),
            paddle.to_tensor([0.2756720781326294], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e0c212cf5c2cc465ad5ae20d8e892179(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f527ad8ab7a192f5752e4c0c71ebf21d
    def get_inputs(self):
        return [
            paddle.uniform([1, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.15963655710220337], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_593c18cf26aca3805b2e4f0719f414c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53d58fd5ff6b85035a89a19e541bb92a
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0.47663119435310364], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_73cd37d7197fb88702efdf50c99169ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c001ff15f1251093e87c7c6cdb02087
    def get_inputs(self):
        return [
            paddle.uniform([300, 6], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.03263209015130997], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_29f7aed30a2ea852cedcfb2af38c88ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 180, 320], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1966366469860077], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9d68a0e6dff93477dce4c91972d0fb5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 256, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.16046729683876038], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_43e43baed453f555fbdac3d5e5c49b1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d51f8a52d6686be09922b630c575fe0
    def get_inputs(self):
        return [
            paddle.uniform([1, 26, 70], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2728694677352905], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7e1b49ca9bfbe55e4a756cdff894de91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 784, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3837188482284546], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8dd7df89b1ef76257b5666537124ddab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d51f8a52d6686be09922b630c575fe0
    def get_inputs(self):
        return [
            paddle.uniform([1, 25, 38], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.0033531691879034042], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_88a02ace1f434f4b2de7d06c05b847df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.193359375]], [[-0.1846923828125]], [[-0.2100830078125]], [[-0.2403564453125]], [[-0.1783447265625]], [[-0.192626953125]], [[-0.201171875]], [[-0.228759765625]], [[-0.2032470703125]], [[-0.151123046875]], [[-0.2396240234375]], [[-0.20556640625]], [[-0.2064208984375]], [[-0.192626953125]], [[-0.2291259765625]], [[-0.2178955078125]], [[-0.153564453125]], [[-0.19970703125]], [[-0.241943359375]], [[-0.1494140625]], [[-0.1805419921875]], [[-0.2113037109375]], [[-0.26220703125]], [[-0.2200927734375]]]], dtype='float16').reshape([1, 24, 1, 1]),
            paddle.to_tensor([0.054565705358982086], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a6a67c1b8670cf9fed6ad2228e38f30d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff15b3b0acf6fd59d587f59e9abf15d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 320], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.42274439334869385], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6f2057c3c1ee340d5261f25432c8fbed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1740708202123642], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bc63b6b2dd0fc4e4ea1cc5632dcff940(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bdc2652643f94045d0f3f6f3e9efa4c
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.5918310284614563]], [[0.8131939768791199]], [[0.771870493888855]], [[0.7580124735832214]], [[0.8121208548545837]], [[0.7120105028152466]], [[0.7660691142082214]], [[0.7746148705482483]], [[0.7948158383369446]], [[0.70881587266922]], [[0.7419081330299377]], [[0.6021258234977722]], [[0.713854193687439]], [[0.7082594633102417]], [[0.7725907564163208]], [[0.7512067556381226]], [[0.624729573726654]], [[0.6152797341346741]], [[0.8247078061103821]], [[0.7464378476142883]], [[0.7393931746482849]], [[0.8544216156005859]], [[0.7520461082458496]], [[0.6818184852600098]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_50b22da2bb50c368abd4ff07d8c837ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0977bdac2b848a6b9560943e7ea74792
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 3, 49, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2404094934463501], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_11cae500a78327fe0d4668e2d18bfd2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff15b3b0acf6fd59d587f59e9abf15d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 180, 320], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.42274439334869385], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fa715a511509a80d57ab4504c018e027(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc9406d6678bff2b2dbc460c1015bbfe
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5af901b3f31e4c8dd8fd2831f081e7f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.14158113300800323], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a54252796b2fc0e68dbef7ae36d320bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce2dfbbd8cf644e56dd9529169710001
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 80, 80, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.46604785323143005], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f2d633579d7f95f90b5a0e291ef738e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff15b3b0acf6fd59d587f59e9abf15d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 80], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.42274439334869385], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0f70d8e3c0211d8edd6b76a0b1db0450(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 180, 320], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.23828814923763275], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_43638f931871180e6457143980a6f20c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4be5717ac1c140ee7f35a2ed285fef48
    def get_inputs(self):
        return [
            paddle.uniform([1, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.26706647872924805], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1c68871c0dc334e97e27c1915a921e13(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('1'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fe58b8ec6f7a2c2f5246e1cff44259f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1c68871c0dc334e97e27c1915a921e13
    def get_inputs(self):
        return [
            paddle.uniform([1, 26, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.26814162731170654], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_97f78b123269b5ec5918cd0c8013c1eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 49, 49], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3803968131542206], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1895b98903e1fd389262381739a7ad98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.33978965878486633]], [[-0.2798660695552826]], [[-0.2910524606704712]], [[-0.2948038578033447]], [[-0.28015658259391785]], [[-0.3072567284107208]], [[-0.2926229238510132]], [[-0.2903095483779907]], [[-0.2848410904407501]], [[-0.30812153220176697]], [[-0.2991633713245392]], [[-0.33700284361839294]], [[-0.3067576289176941]], [[-0.3082721531391144]], [[-0.2908574938774109]], [[-0.29664620757102966]], [[-0.3308839201927185]], [[-0.3334420323371887]], [[-0.27674925327301025]], [[-0.29793715476989746]], [[-0.2998441755771637]], [[-0.2687056362628937]], [[-0.2964189946651459]], [[-0.31542980670928955]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor([0.14158113300800323], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4918b8703cb20af9c0890cb770104c8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0977bdac2b848a6b9560943e7ea74792
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 80, 80, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.05878882110118866], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_238c15b331cd7830108c885cdb3396e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1aa6af613065edb17e05380c4ad3e6dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1769556850194931], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_41b6fd558319596c82465f93e0c4b335(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a38cf865df5cfd3b84f4bae48ed2ab7d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.0146636962890625]], [[-0.01531982421875]], [[-0.01251983642578125]], [[-0.01336669921875]], [[-0.01213836669921875]], [[-0.01485443115234375]], [[-0.013458251953125]], [[-0.0142822265625]]]], dtype='float16').reshape([1, 8, 1, 1]),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_37d2374f47d7acc361dac7c14c8289d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bdc2652643f94045d0f3f6f3e9efa4c
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a6788d05f9fc776015105ce5ad9f4dc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([16, 8, 144, 32], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.14657902717590332], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_65e6ad80db46d0d0693a1178e8e97cf5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 96, 160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4670775532722473], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_638688f7c11f6f4a3b624106e299be55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0977bdac2b848a6b9560943e7ea74792
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 20, 20, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2573584020137787], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ef70bb1173eea0052e60cd5bf33e25f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 196, 196], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.13168077170848846], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bc6a3060da5b4f1c01875e12f13f70ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc9406d6678bff2b2dbc460c1015bbfe
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.8662109375]], [[0.890625]], [[0.81884765625]], [[0.7333984375]], [[0.90869140625]], [[0.8681640625]], [[0.84423828125]], [[0.76611328125]], [[0.83837890625]], [[0.9853515625]], [[0.7353515625]], [[0.83154296875]], [[0.8291015625]], [[0.8681640625]], [[0.76513671875]], [[0.796875]], [[0.978515625]], [[0.84814453125]], [[0.72900390625]], [[0.990234375]], [[0.90234375]], [[0.8154296875]], [[0.67138671875]], [[0.79052734375]]]], dtype='float16').reshape([1, 24, 1, 1]),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_449d724d395df01ffa7a409e0989d5e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f527ad8ab7a192f5752e4c0c71ebf21d
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4670775532722473], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c4e13772c4452f336df0787950a49813(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 49, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.016803564503788948], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_99c5f7ba5218aae161b39ce175260836(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_44c5345e9eb03d03bca15d756b448745(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.13788749277591705], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_60836bcc4246a9dfa59473a7246e46c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a38cf865df5cfd3b84f4bae48ed2ab7d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.01204681396484375]], [[-0.007965087890625]], [[-0.01189422607421875]], [[-0.007965087890625]], [[-0.007965087890625]], [[-0.01058197021484375]], [[-0.01175689697265625]], [[-0.01242828369140625]], [[-0.007965087890625]], [[-0.01273345947265625]], [[-0.00946807861328125]], [[-0.008758544921875]], [[-0.01068115234375]], [[-0.01027679443359375]], [[-0.00921630859375]], [[-0.00937652587890625]]]], dtype='float16').reshape([1, 16, 1, 1]),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_742ebf8c44524c92dc5311c5e3b8ad15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce2dfbbd8cf644e56dd9529169710001
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 20, 20, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.13843213021755219], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6dc52f14c5b4d823c79142a43083d9ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aff2cb8292404abf8b5af6bddb65de41
    def get_inputs(self):
        return [
            paddle.uniform([3200, 20], dtype='float64', min=0, max=0.5),
            paddle.to_tensor([0.266884982585907], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_430b470b40ea2d4002ec651224c41d03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f265e95a70bb268fbda1f9b746cd173
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a451fc54b2f695ea4aa4896a3b343242(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('-0.5'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ca09ed254fc9173fb601c3b46194748a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a451fc54b2f695ea4aa4896a3b343242
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 80, 80, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.22985711693763733], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e5987a82986400192620fc7a74ef871c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 784, 49], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.15890726447105408], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2b12180c578d54bf6eeb34cdca594fd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c001ff15f1251093e87c7c6cdb02087
    def get_inputs(self):
        return [
            paddle.uniform([1, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.009423283860087395], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c3adb2ecbb7a425e6e6700ecaec09fb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 49, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2251664251089096], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2527af7f0a837682f0484d707180d4cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 350, 25], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.07456143200397491], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2757c0ddb8041bef2146a91a528a3fec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0977bdac2b848a6b9560943e7ea74792
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 40, 40, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.05878882110118866], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8c15a190d052b3e57d980c2c571d16f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6f0b325e939538df2031df44fc1d78c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f527ad8ab7a192f5752e4c0c71ebf21d
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.12629848718643188], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cbe5129cbe8c9bbbbea3a2a5b4cd523f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 144, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.27207040786743164], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3f3c8ecd4ea63f284b0d942ac41faacd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d51f8a52d6686be09922b630c575fe0
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 320], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3062542974948883], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e814ef2ea8addbc4e8b5d301f9697a90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 196, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.446737140417099], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0453129c5e5c82ae9f0c4022fb3f0353(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 49, 49], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.49470216035842896], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_642ff268dbe0d1031d2afd27a96071a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d51f8a52d6686be09922b630c575fe0
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 320], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2011336237192154], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_732aa00cefa1d51fd2887b48411f562e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 3136, 49], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.41277825832366943], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_143a0684f8541760549b233a256e1e0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f527ad8ab7a192f5752e4c0c71ebf21d
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.06225132569670677], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8d352685473e209c099cc247a2e45c51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f527ad8ab7a192f5752e4c0c71ebf21d
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.49733033776283264], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_3e30b05d516aa8d96da8f796ea6e147c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('-1'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_42279b578808bcda913e65c66c81b12d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e30b05d516aa8d96da8f796ea6e147c
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0021241477224975824], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f9d5e9d9a2600cedf7829a1171986ca3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a38cf865df5cfd3b84f4bae48ed2ab7d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.01055145263671875]], [[-0.01007843017578125]], [[-0.01146697998046875]], [[-0.01311492919921875]], [[-0.009735107421875]], [[-0.0105133056640625]], [[-0.01097869873046875]], [[-0.012481689453125]], [[-0.0110931396484375]], [[-0.00824737548828125]], [[-0.0130767822265625]], [[-0.0112152099609375]]]], dtype='float16').reshape([1, 12, 1, 1]),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_204110bcc7cd7b84337450a0634eedc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f527ad8ab7a192f5752e4c0c71ebf21d
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.15963655710220337], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0762e0233f47c41791041b95ff08bc45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6e3a80ed59e509c06c0ea7261bbc2060(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([2, 32, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1769556850194931], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_98219d596a9db3d7ff0fb3bd3762073b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 49, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17822237312793732], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_22ff552ba520033d0cb743c2628117b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.14158113300800323], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5cc35788f4fbae6df03be43b11b64460(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0977bdac2b848a6b9560943e7ea74792
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 80, 80, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2297879308462143], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f0a38b13d624153a180372a4fa49a780(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f527ad8ab7a192f5752e4c0c71ebf21d
    def get_inputs(self):
        return [
            paddle.uniform([1, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.16184796392917633], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eac9ac65b0322b5273db2453ef89af1d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f527ad8ab7a192f5752e4c0c71ebf21d
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.345152884721756], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_67e4f24c0967c00c5ea44e6bc5a8f2e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.01454925537109375]], [[-0.01369476318359375]], [[-0.01424407958984375]], [[-0.01230621337890625]], [[-0.012542724609375]], [[-0.01416015625]], [[-0.01334381103515625]], [[-0.01316070556640625]]]], dtype='float16').reshape([1, 8, 1, 1]),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_da5fbf744ff8e5832c622ff8988bfa1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1aa6af613065edb17e05380c4ad3e6dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.21931400895118713], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d9939c311b059bab0891dc43eebe8149(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d51f8a52d6686be09922b630c575fe0
    def get_inputs(self):
        return [
            paddle.uniform([1, 180, 320], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.0881076231598854], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9708aa435e0eb96f6cfc9b64a94d5084(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1aa6af613065edb17e05380c4ad3e6dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1769556850194931], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2625fd4d5b009f8b7cf47eba83c55c44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f527ad8ab7a192f5752e4c0c71ebf21d
    def get_inputs(self):
        return [
            paddle.uniform([1, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.21931400895118713], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_360eba72f7824d8af082602cc2f287fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.14158113300800323], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c7a9f771449557f938354eea2a05b464(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1fecfbf9f8d004aa242d9989f36ef751
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 40, 40, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.03263209015130997], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_308ca8e8f6c5535bf85fef972c28bc4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1aa6af613065edb17e05380c4ad3e6dc
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.21931400895118713], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_830fc48630842c1d93425bbd36ec1d65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a451fc54b2f695ea4aa4896a3b343242
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 40, 40, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.22985711693763733], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7e19fc02e35c57d7aafaca82228bf6d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2763035297393799], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b147ffbf08b83df5a8b9a5c768248b9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.0120391845703125]], [[-0.01102447509765625]], [[-0.009490966796875]], [[-0.00812530517578125]], [[-0.01200103759765625]], [[-0.007965087890625]], [[-0.0115509033203125]], [[-0.009307861328125]], [[-0.007965087890625]], [[-0.0091705322265625]], [[-0.00826263427734375]], [[-0.01119232177734375]], [[-0.00855255126953125]], [[-0.00807952880859375]], [[-0.008056640625]], [[-0.007965087890625]]]], dtype='float16').reshape([1, 16, 1, 1]),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_199a63dc381143f21b6e3edaa4d6814c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.054565705358982086], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_756da6ecec1a78b4444b85a2f4cc3109(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f527ad8ab7a192f5752e4c0c71ebf21d
    def get_inputs(self):
        return [
            paddle.uniform([1, 26, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.02612169273197651], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_077586ab6d6a809ca0a1ee935c672312(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ea32138c0e3f77846751b20649429ba3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0977bdac2b848a6b9560943e7ea74792
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 40, 40, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.39329299330711365], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b2924a6eaf14dd9455c69025a95d9028(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f527ad8ab7a192f5752e4c0c71ebf21d
    def get_inputs(self):
        return [
            paddle.uniform([1, 26, 70], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0021241477224975824], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ddf271fcd4debd5cc395578d1576f2d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 49, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2404094934463501], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0d2519ab072f59a1032394c32438aae0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bdc2652643f94045d0f3f6f3e9efa4c
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dfa385a2f878ec77ad67979d4af81ee8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.04343109205365181]], [[-0.04364551976323128]], [[-0.04117993265390396]], [[-0.04199950769543648]], [[-0.046846918761730194]], [[-0.04720909893512726]], [[-0.039182472974061966]], [[-0.042182281613349915]], [[-0.04245227947831154]], [[-0.03804364800453186]], [[-0.04196733608841896]], [[-0.04465891048312187]]]], dtype='float32').reshape([1, 12, 1, 1]),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_55e551adf878f829a04d9a77c06c0ed6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 49, 196], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.13168077170848846], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dac707298bb7b6c55d28e34528468465(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 192, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.15963655710220337], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_975bee15d2573ad067fc9a6f85de62d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cc9406d6678bff2b2dbc460c1015bbfe
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_247e7612362bcc1e8875795d5c8b7eab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f527ad8ab7a192f5752e4c0c71ebf21d
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 30], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.12629848718643188], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8fd37dc0cef0727957041a30e786c791(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d51f8a52d6686be09922b630c575fe0
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 4], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.37192127108573914], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_022710352f1a809f171dd7be9c62c647(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.054565705358982086], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9adbd1fb8dad5b53c648aecf394f3db8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bdc2652643f94045d0f3f6f3e9efa4c
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a15b9fa9fb7646e2e398ad35f2b31bda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce2dfbbd8cf644e56dd9529169710001
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 6, 49, 49], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.41277825832366943], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fab52305f91d7c9aefbec3cb8700deb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 784, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.016803564503788948], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4d6379ecd62146b4724d12b9026e8ed3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce2dfbbd8cf644e56dd9529169710001
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 20, 20, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.15257292985916138], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eb152b8d9f77251068bcec8c3ef89c58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f527ad8ab7a192f5752e4c0c71ebf21d
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.31343406438827515], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_67008658336b515a58b36c977647318f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce2dfbbd8cf644e56dd9529169710001
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 40, 40, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.13843213021755219], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1592741134ae9cfe604c1108c1b6a700(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f265e95a70bb268fbda1f9b746cd173
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.03653030842542648]], [[-0.03514935448765755]], [[-0.04353402554988861]], [[-0.039983395487070084]], [[-0.037438008934259415]], [[-0.04565148428082466]], [[-0.04420868679881096]], [[-0.04190097376704216]], [[-0.03843735158443451]], [[-0.037291187793016434]], [[-0.041884440928697586]], [[-0.03599511831998825]], [[-0.03529391810297966]], [[-0.040505290031433105]], [[-0.04184701666235924]], [[-0.04465116932988167]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_298b89a2ebc9a853420beaea3b8b939e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.268798828125]], [[-0.28076171875]], [[-0.2294921875]], [[-0.2449951171875]], [[-0.222412109375]], [[-0.272216796875]], [[-0.2467041015625]], [[-0.26171875]], [[-0.2666015625]], [[-0.2509765625]], [[-0.260986328125]], [[-0.2254638671875]], [[-0.2298583984375]], [[-0.259521484375]], [[-0.2445068359375]], [[-0.2412109375]]]], dtype='float16').reshape([1, 16, 1, 1]),
            paddle.to_tensor([0.054565705358982086], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8be37c44fc9c6c05142d59a834019bb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f527ad8ab7a192f5752e4c0c71ebf21d
    def get_inputs(self):
        return [
            paddle.uniform([1, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1769556850194931], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cc0bc78f39a8eb99f835d56a222e6282(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d51f8a52d6686be09922b630c575fe0
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 80], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.28113576769828796], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2e95bdf02ea1ac698721c58f3337f720(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f527ad8ab7a192f5752e4c0c71ebf21d
    def get_inputs(self):
        return [
            paddle.uniform([1, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.266884982585907], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8d2ec761eae661d51278b63dc455ac1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce2dfbbd8cf644e56dd9529169710001
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 20, 20, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.46604785323143005], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1f82ff14d6261c62627cb4219e1ff670(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce2dfbbd8cf644e56dd9529169710001
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 24, 49, 49], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.41277825832366943], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1461ed1f524d0da99712a64fd9efa0cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4be5717ac1c140ee7f35a2ed285fef48
    def get_inputs(self):
        return [
            paddle.uniform([300, 6], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.22985711693763733], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1faccf7ed02e1fd7486e1bf769be68fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d51f8a52d6686be09922b630c575fe0
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 80], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.09170089662075043], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c1bea3b85fe85923a3f392eec46cae45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f265e95a70bb268fbda1f9b746cd173
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.046710580587387085]], [[-0.0435403548181057]], [[-0.03985964506864548]], [[-0.041160374879837036]], [[-0.04314696416258812]], [[-0.04647710174322128]], [[-0.03648686781525612]], [[-0.04088347405195236]]]], dtype='float32').reshape([1, 8, 1, 1]),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_584151ad91e91a670e7b5f086872a40f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a38cf865df5cfd3b84f4bae48ed2ab7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2c2a5d55b838690584d53cb2c91d412d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f527ad8ab7a192f5752e4c0c71ebf21d
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.01980632171034813], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1825a44fb0b208c7889b693e1d72e9ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d51f8a52d6686be09922b630c575fe0
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 30], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.37192127108573914], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d6a1b1bca895a83fd52031f9bc13b4f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f527ad8ab7a192f5752e4c0c71ebf21d
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.16694828867912292], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_18a6f3a5d7d26c678e712a11f4a3d43c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d51f8a52d6686be09922b630c575fe0
    def get_inputs(self):
        return [
            paddle.uniform([1, 180, 320], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.42274439334869385], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1d4f749b726e845cc18c371dd9ebe17a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.0443836934864521]], [[-0.04300510883331299]], [[-0.032996173948049545]], [[-0.03791964054107666]], [[-0.043183378875255585]], [[-0.0381709523499012]], [[-0.037291646003723145]], [[-0.03863013908267021]]]], dtype='float32').reshape([1, 8, 1, 1]),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_352046c36226e7d8e3442c1c16323fce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a38cf865df5cfd3b84f4bae48ed2ab7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2728694677352905], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4c9b2f136014a90c39d3f2d750a1f71d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f527ad8ab7a192f5752e4c0c71ebf21d
    def get_inputs(self):
        return [
            paddle.uniform([1, 2125, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.43221747875213623], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1067ac012885b36e9f7eb78d06aea358(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([2, 32, 180, 320], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.42274439334869385], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6d28815596454b5e1f13c7400a30d4b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.054565705358982086], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_afc62de292107dc346124ffb6a7c0dca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.054565705358982086], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_65b842f292a4d64acf3a7e63f80f84a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 784, 49], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.41277825832366943], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_21b9e40d03b604ab4e07e60f028f5d49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 128, 128], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.15648093819618225], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8926a65492e7eb60e32659d0882d0c48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0977bdac2b848a6b9560943e7ea74792
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 20, 20, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.05878882110118866], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_42b28ea739dcd6bf0602736b6f7209ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a451fc54b2f695ea4aa4896a3b343242
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 20, 20, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.22985711693763733], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_948ac43143d2ce2a616558d9651d59c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d51f8a52d6686be09922b630c575fe0
    def get_inputs(self):
        return [
            paddle.uniform([1, 180, 320], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4322897791862488], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_777dabfb5a68c16c9d8187a1f3994418(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 3136, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2404094934463501], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_62efb87601e58d98e2a4cdf3c585756d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.14158113300800323], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_52222b25753c87de63b18a5f6fa66810(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d51f8a52d6686be09922b630c575fe0
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3962007164955139], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8508718d333c7efc038c548b7aa703e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1bdc2652643f94045d0f3f6f3e9efa4c
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9011cd6fc8c1d0bbe7d59601bf52eed1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ce3fa442cbff2335327575d1cd82c78
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 49, 49], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.41277825832366943], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d911de7d4568fabe028cf32183b1dc01(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f265e95a70bb268fbda1f9b746cd173
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_054521f9334e2523e9a1336ab2db3c33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 96, 160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.15963655710220337], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8832f32d3c74c44f9cb5d1b470e6901e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3cd8679d0ec572445b61cfe87b9e6598
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.039256930351257324]], [[-0.04020240902900696]], [[-0.04218309000134468]], [[-0.038093797862529755]], [[-0.03595396876335144]], [[-0.03911122307181358]], [[-0.03731000795960426]], [[-0.04295990243554115]], [[-0.040298208594322205]], [[-0.046923860907554626]], [[-0.04241222143173218]], [[-0.034252773970365524]], [[-0.04272951930761337]], [[-0.044259000569581985]], [[-0.03523772582411766]], [[-0.03425834700465202]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_37f7ba160c81e7cb1cc75f66afd22b52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aff2cb8292404abf8b5af6bddb65de41
    def get_inputs(self):
        return [
            paddle.uniform([3200, 20], dtype='float64', min=0, max=0.5),
            paddle.to_tensor([0.0021241477224975824], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_73d181f03800b481a6c9e99497215a69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ce2dfbbd8cf644e56dd9529169710001
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 80, 80, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.13843213021755219], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6299563c4231b12c49e4fb630e4a3c30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f527ad8ab7a192f5752e4c0c71ebf21d
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3747236430644989], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c1c6ba1df37a963f15f9cbcb9d8ef9c5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fd641051e798858d14125bac765b959c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1c6ba1df37a963f15f9cbcb9d8ef9c5
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.054565705358982086], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ef938ac95e95e4cb92eff1f95135e2d7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, 256, 256], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_41bdcf76fb9eeaca0e4773d258e02885(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef938ac95e95e4cb92eff1f95135e2d7
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 256, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2397179752588272], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_632b62e5df2400d33022cd0f6671a36d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 16, 16], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f6a2e1feefa39fb5be2a32fd750d0318(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_632b62e5df2400d33022cd0f6671a36d
    def get_inputs(self):
        return [
            paddle.uniform([196, 4, 16, 16], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4900263249874115], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_612d5452deea53476be4c8ebc45a6a45(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 5, 196, 49], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_512d5d017c412f6db4633358d9268b52(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_612d5452deea53476be4c8ebc45a6a45
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 196, 49], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.15890726447105408], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1aa503c45f4a343283f38209d9c856fe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 80, 80, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3f5feed7c8007be93db19f672c2788ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1aa503c45f4a343283f38209d9c856fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 80, 80, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.27328571677207947], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b9a58eda3c7f4e9e4e2a976d30ad777a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('-0.5'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ffd556c1f1bf1427ddc4d727aff65e55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9a58eda3c7f4e9e4e2a976d30ad777a
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5272a72f3031ccdcf287936b1b4a22ed(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 40, 40, 2], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_43f64a45768aab1cb811832dc5e1aa84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5272a72f3031ccdcf287936b1b4a22ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 40, 40, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.46604785323143005], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e83b267e1e19f3fda3d4c90a7ae46765(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2, 56, 24], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3708e3e7f41215c73862bc6dbae98b1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e83b267e1e19f3fda3d4c90a7ae46765
    def get_inputs(self):
        return [
            paddle.uniform([56, 2, 56, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2251664251089096], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_068a02d70e2e69f0bea0b065ef0034be(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, 26, 26], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_be38f9998eb2d4453b7d5a24707ef592(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_068a02d70e2e69f0bea0b065ef0034be
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 26, 26], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2397179752588272], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4f92820ea8f483dcf6433fe3d5393bcc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 5, 196, 49], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_34a781140c6286895c97ac523901327e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f92820ea8f483dcf6433fe3d5393bcc
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 196, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3837188482284546], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4340d7da94a3bfb1b9187e04147e04e7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('-0.5'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2db823b1f3fd721754a4200156f34c37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4340d7da94a3bfb1b9187e04147e04e7
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_fbad0f25407e7f1df86bd70f6bca076d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 144, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1f2f51455395f93dcb4a8f3b46c7c128(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fbad0f25407e7f1df86bd70f6bca076d
    def get_inputs(self):
        return [
            paddle.uniform([4, 16, 144, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.27207040786743164], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9efe977f3774294c446c4ffedf6f5752(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 25, 25], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_912c26d8714e28f9420aeb4d91358f1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9efe977f3774294c446c4ffedf6f5752
    def get_inputs(self):
        return [
            paddle.uniform([1, 25, 25], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.34655219316482544], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_73d9f6c89d225cfceab5988cd19c21a4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('1'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_95c0aa3be87292f8fc71f4b7ecdce2ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73d9f6c89d225cfceab5988cd19c21a4
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5cbdbc2f5d22cd9c6f8fe4968ca732ea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, 26, 26], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_68eea0527e628c755374f0e60f736bf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5cbdbc2f5d22cd9c6f8fe4968ca732ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 26, 26], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.08247071504592896], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_70f1e91dd3af992d9a268f80b8d72121(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('1'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f9aed57378067cb1fad8a0dcf22f69ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_70f1e91dd3af992d9a268f80b8d72121
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_86449f920e51cef1d159ca6b18954e42(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6e36bc26f80d096ffe0bee4eb3a8dc1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86449f920e51cef1d159ca6b18954e42
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.054565705358982086], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4ae4111d2d43c157e2565a8686ef976b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, 98, 24], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5d1f6f641c05e2897cbb79da76db7655(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ae4111d2d43c157e2565a8686ef976b
    def get_inputs(self):
        return [
            paddle.uniform([2, 8, 98, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17822237312793732], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d5ba742bdf3e506ad4362f521e425d78(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 144, 180, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_06cb385afed19052b751f4e5140e94be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5ba742bdf3e506ad4362f521e425d78
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4858695864677429], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4d5bdb6e51010f4032d96234f5df761a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 6, 197, 197], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5109693d0a1068b9d5e713ebff7b1177(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d5bdb6e51010f4032d96234f5df761a
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 197, 197], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2784574031829834], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_49a6526d70e907b1fe4a1aa08d066abd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f0a550884f5b1fdad6c947b7695efcf5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_49a6526d70e907b1fe4a1aa08d066abd
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_cda2b26a5d299666dc4c9dbab0c2246a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 12, 197, 197], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c46644b75b957c62100ee6c2f8d3ddb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cda2b26a5d299666dc4c9dbab0c2246a
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 197, 197], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3810800313949585], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_022aba8c0db6819a2445c331aabe732c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, 98, 24], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_344e254fd2e7b81c5552d11f534958a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_022aba8c0db6819a2445c331aabe732c
    def get_inputs(self):
        return [
            paddle.uniform([2, 8, 98, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2251664251089096], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_feced3934b818051896a4e421906a9c4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 5, 196, 196], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_76927e1c750abb34040366453863a64a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_feced3934b818051896a4e421906a9c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 196, 196], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.039581477642059326], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4cd7040986cfc7294f1c77fcf4216e8a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 32, 180, 320], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_abefdfa2914f3901f032a464ddbb362d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4cd7040986cfc7294f1c77fcf4216e8a
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 180, 320], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.42274439334869385], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_554327c6a1cc6d5e73b08d984078aa03(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 56, 24], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3053850e9ea1728c988e95dea5e9cdad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_554327c6a1cc6d5e73b08d984078aa03
    def get_inputs(self):
        return [
            paddle.uniform([14, 4, 56, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17822237312793732], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e62df87ed2fc6ef644445ccb85e29fc9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('1'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fbee30df782960aaea22e600007ec956(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e62df87ed2fc6ef644445ccb85e29fc9
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8a8db6d7dca929d2b3e25f2941011704(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_68dde3ef458251c35fedef863844d598(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a8db6d7dca929d2b3e25f2941011704
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.21983124315738678], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_058d6b68c4b80447dd5048279a61bd5c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 12, 196, 49], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dd47704219edada4e3a8aaa9ff2e196e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_058d6b68c4b80447dd5048279a61bd5c
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 196, 49], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.41277825832366943], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_922c9842dec687687567dafea6d65cd7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, 49, 49], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_448723d00d22f1fc0fbd3d255fec1a26(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_922c9842dec687687567dafea6d65cd7
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 49, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.27348795533180237], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_3dd3e2336f9e1ae1d0d594882c5f5f88(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c8e7318a0ec637e5c1ee57dde72a8e5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3dd3e2336f9e1ae1d0d594882c5f5f88
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_46e6e0d9aa06e99992676f9bd164a77e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_922c9842dec687687567dafea6d65cd7
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 49, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3837188482284546], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_3178b76fe216295bf0d228d3788783cc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, 49, 49], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5da86d94e3277803d178735fe4644912(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3178b76fe216295bf0d228d3788783cc
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 49, 49], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.15890726447105408], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_519af1151b8858cafe172b6684ee2310(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 192, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a7bbf59b73e96fc499d2992ced49599c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_519af1151b8858cafe172b6684ee2310
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 192, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4670775532722473], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e07fa4e02b45189c3c3bbfc44da0a28b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 144, 32], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_566ef3b20552cac4396ea06e7771ef9f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e07fa4e02b45189c3c3bbfc44da0a28b
    def get_inputs(self):
        return [
            paddle.uniform([4, 16, 144, 32], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.14657902717590332], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_fc84fb4115250adeb7c266147da2c815(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 180, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_90dda4fd153c3f3b6720918db7c0e582(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc84fb4115250adeb7c266147da2c815
    def get_inputs(self):
        return [
            paddle.uniform([1, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.49733033776283264], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f942792004635b32d9f26fdf066c0696(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6f710142aac527b26a2903c82135f005(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f942792004635b32d9f26fdf066c0696
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9640511b40c2f8ffdce036774be20147(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 256], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0f6acd5ab2cf35fbf352dadf8348aa81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9640511b40c2f8ffdce036774be20147
    def get_inputs(self):
        return [
            paddle.uniform([8, 256, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4651494026184082], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4f7eeb8b041d46c0c738f01729e46539(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('-0.5'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 768, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1d107b4c1c7f7ffbd76c9736d6045675(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f7eeb8b041d46c0c738f01729e46539
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_185f25402617612009ff599bd65c47c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f92820ea8f483dcf6433fe3d5393bcc
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 196, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.016803564503788948], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_26be76a37384478f3c6f76e308f240a1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 32, 180, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_69cd2cc848aff58a3711204a478c04fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_26be76a37384478f3c6f76e308f240a1
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1769556850194931], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8f6586ff6d0cb9c1d00de43ee38e8836(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 16, 60], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1a88df36e464cd41f2fed6d81704147b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f6586ff6d0cb9c1d00de43ee38e8836
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 16, 60], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4147603213787079], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4a73c96ffb0dacdc6e5c6c5f95bf0f22(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_615d3b0c1f606b44c8473692e579a42e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a73c96ffb0dacdc6e5c6c5f95bf0f22
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0d7b392f6130edf34f4954160f48362b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3178b76fe216295bf0d228d3788783cc
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 49, 49], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.09803877770900726], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6f85f1dc8bce4c70ecb15d27de6e70a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 12, 577, 577], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8d3adc28710c8ca37fcd9c70535a936b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f85f1dc8bce4c70ecb15d27de6e70a6
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 577, 577], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.22600552439689636], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e9878f8bb0cb5b468123b76dcd7f188c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 48, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ed5cdf37380eca901865028d546aa2b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9878f8bb0cb5b468123b76dcd7f188c
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4322921633720398], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b3eb00949617b37c0f23b6f80426c373(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8bf8a2911b287943c04228a91e7ca2f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3eb00949617b37c0f23b6f80426c373
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_65895e30edac3b66e6ec3d59aa965360(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a73c96ffb0dacdc6e5c6c5f95bf0f22
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.054565705358982086], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0207c321033e503ef5fb50f6fbf9efa7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ab717101569bb9347b98403436d9cf1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0207c321033e503ef5fb50f6fbf9efa7
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.14158113300800323], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b3a3435deb3e21c76f9a61b72a31d461(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 3136, 49], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_01555f13c3f38971c5d0d00e4d6d8a83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3a3435deb3e21c76f9a61b72a31d461
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 3136, 49], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2799938917160034], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5dd6086c6e02c2b8722f89633ceff045(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('-1'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 160], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8903fe7ebdcdfd60c85145099f47a770(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5dd6086c6e02c2b8722f89633ceff045
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 160], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.42274439334869385], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_871556bbbd3fa4e4bf71570ffb10d4e4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 501, 30], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e1cc78b3652eaae924049708d0dedfe4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_871556bbbd3fa4e4bf71570ffb10d4e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 30], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.18577031791210175], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_aabe363301b62c17998b5d884896cca1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('-0.5'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c3ee3701f56c6baa07721ff76953dd45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aabe363301b62c17998b5d884896cca1
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f9c999e3ce379e6b49e810cf629b5d36(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 5, 196, 196], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0e4d7b1e737dc98365ce1c08855b7c23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9c999e3ce379e6b49e810cf629b5d36
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 196, 196], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.09803877770900726], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_27fd950d7fec4e2522dd3ad280fb61ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a8db6d7dca929d2b3e25f2941011704
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.15963655710220337], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fac1d0bb82d1dd04fbd5cd9d6e93e8b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a8db6d7dca929d2b3e25f2941011704
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3119860291481018], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f46fe936553ff2f9616ffc4a9aa200d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cda2b26a5d299666dc4c9dbab0c2246a
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 197, 197], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.35620564222335815], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_efa756db7d374f0cd89cc73f415282ab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('-0.5'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a724041beddd6356cd807b1b432b438a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efa756db7d374f0cd89cc73f415282ab
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d9e5161e7c922b956949852bb769e231(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 160], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2fb0352d0103747431443735be33fda9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9e5161e7c922b956949852bb769e231
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4670775532722473], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f435cf632a2167f6a7b4aceeebcf702d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 12, 197, 197], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_642b5b0f6a27c49428c3a1264b03cf6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f435cf632a2167f6a7b4aceeebcf702d
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 197, 197], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.44839635491371155], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d3591d0a242cb9eedaf11a1b1b7114ce(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fe0de15b99b5f15caed47d4a0a7dd7b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3591d0a242cb9eedaf11a1b1b7114ce
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b3d20abffd764d95bceb7d4bbae9d837(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9efe977f3774294c446c4ffedf6f5752
    def get_inputs(self):
        return [
            paddle.uniform([1, 25, 25], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4551265835762024], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b2a8c6021db559dfe726fa0a16fd97d7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 40], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bb1e2a6643c8565a735409950e6d12dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2a8c6021db559dfe726fa0a16fd97d7
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0.016806144267320633], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_11189ff9f1db6b3632bd6c7160d961a1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('1'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 384, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_deb1df271a0fe92d8eb509aaa5853632(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_11189ff9f1db6b3632bd6c7160d961a1
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b1a18afec680549793416f04e7a7ac59(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1024, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9ef68a4ef82ea5c2f248d9360c0837a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b1a18afec680549793416f04e7a7ac59
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.14158113300800323], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c74b7c9381f08a4696d9ab58582db131(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0207c321033e503ef5fb50f6fbf9efa7
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1029fb76832821d17016697d67418c8a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('1e-06'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[3200, 20], dtype='float64'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_28bd3ebf73051ca8c7bd09108c45a817(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1029fb76832821d17016697d67418c8a
    def get_inputs(self):
        return [
            paddle.uniform([3200, 20], dtype='float64', min=0, max=0.5),
            paddle.to_tensor([0.0033531691879034042], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_713a638bc34d8e2fdccb6a632b259c1c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 72, 180, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a59b23ada949cbee94ff09bd1eed54a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_713a638bc34d8e2fdccb6a632b259c1c
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4858695864677429], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6e6c123416e3a22e5d7c1aa7c8d7ecf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f9c999e3ce379e6b49e810cf629b5d36
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 196, 196], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3803968131542206], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_123a5058e91b58091505e5c554c456c9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('1'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_40a557617147727760c7f79458b2d0dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_123a5058e91b58091505e5c554c456c9
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_88bcfbaea4f56705846a72bcf775955c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_30869cb07cce59083de3784a5d34e6cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88bcfbaea4f56705846a72bcf775955c
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.21536855399608612], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9a67902e8a00a813dace50119c562c00(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 160], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_81f26288c73b8c3ba14937b2acd8ac7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a67902e8a00a813dace50119c562c00
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 160], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2011336237192154], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_383174650404bd2e9eabfddfd849e753(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('-0.5'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_148e0a5df65bbffc3f5d0ee7cfd2a6ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_383174650404bd2e9eabfddfd849e753
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2ee0aadc20b939c5815312a4e8587c52(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 80, 80, 2], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eb12eb461b92afeae5d21cb0b29b36ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ee0aadc20b939c5815312a4e8587c52
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 80, 80, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3051380217075348], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8f1df4b947632cf22c931003f3a6dd7d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('-1'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 180, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_85c5e17ae7876a1fb10ccdbd58af4469(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f1df4b947632cf22c931003f3a6dd7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1769556850194931], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_342b6ff08263429be83bd6fbced840b9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 38], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_35f5405e7ebda28e46c028a1a99f7a58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_342b6ff08263429be83bd6fbced840b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 38], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2756720781326294], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_da31a2d39f60fe446c00dd296f204b36(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('1'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 12, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0efefaa4ec938ef8945a9b72213a5fad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da31a2d39f60fe446c00dd296f204b36
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.0481078065931797]], [[-0.03962375596165657]], [[-0.041207537055015564]], [[-0.041738662868738174]], [[-0.03966488689184189]], [[-0.043501757085323334]], [[-0.04142988473176956]], [[-0.04110235348343849]], [[-0.040328122675418854]], [[-0.04362419620156288]], [[-0.04235588759183884]], [[-0.04771324619650841]]]], dtype='float32').reshape([1, 12, 1, 1]),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d5e40a6343ca3d9d45336185e5ebfbc3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 48, 80], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3de6aeaecf4f8a86d1651e67c5b9d420(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5e40a6343ca3d9d45336185e5ebfbc3
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 80], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2011336237192154], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_caa5c43ac9c047133bcad755afc7155b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 3136, 49], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d62c3263d5facefabc4185b40939213a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_caa5c43ac9c047133bcad755afc7155b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 3136, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.016803564503788948], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6d0bf5f24ee0f4c25169b3520aac8ca9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 25, 38], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_24572f9898f2533293224b7e7624c572(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d0bf5f24ee0f4c25169b3520aac8ca9
    def get_inputs(self):
        return [
            paddle.uniform([1, 25, 38], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2756720781326294], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d8f9e02b1254042567ddcb025602644a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 3, 49, 49], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5c6982bb60efda7afd11aac182323b0c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8f9e02b1254042567ddcb025602644a
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 3, 49, 49], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.41277825832366943], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_409e5e4b343cf4ccc512e0686e2e1f03(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 24, 49, 49], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_68b1c190c0da0a6084b157687343eae2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_409e5e4b343cf4ccc512e0686e2e1f03
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 49, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.446737140417099], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9959e5c78bca92005cb456ca8ebea07c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4c077ca51a98dc471ad84aeaee91bbcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9959e5c78bca92005cb456ca8ebea07c
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4d85c129d95138bb04e32d0e78889b1f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 6, 197, 197], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cad38b9e9b979764a47164860331f9e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d85c129d95138bb04e32d0e78889b1f
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 197, 197], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4892241060733795], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_27e8314fe0a8a72fbdcea81c429433c6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('1'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f4c6978dde913e0ea4e6e0618310912b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_27e8314fe0a8a72fbdcea81c429433c6
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_81cc2ec7b4f68d2531f14261eea50435(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 25, 25], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0361c95eac73e3741be7ff667e50a21f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81cc2ec7b4f68d2531f14261eea50435
    def get_inputs(self):
        return [
            paddle.uniform([1, 25, 25], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2309761345386505], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ba4dfdd5d38b89a88e72f81c8afd60c2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 6, 784, 49], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7f13fc8e8a0c5f67c016da5dee6eb20e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba4dfdd5d38b89a88e72f81c8afd60c2
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 784, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2404094934463501], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d5d6eeca39dc9f210375d789315e531b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('1'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6f4dcc52f9edbe6dc66b29c7575d34ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5d6eeca39dc9f210375d789315e531b
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ea981a635877f28ca449eb52dda09615(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1029fb76832821d17016697d67418c8a
    def get_inputs(self):
        return [
            paddle.uniform([3200, 20], dtype='float64', min=0, max=0.5),
            paddle.to_tensor([0.22651496529579163], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_74743677b5be2f04a90ec8c5db4ef91b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_922c9842dec687687567dafea6d65cd7
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 49, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.039581477642059326], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c74cc7e2548ee0e8712055f4be7f7888(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('-0.5'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5a13f9bd8ab34bdc706ccd16d3d658cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c74cc7e2548ee0e8712055f4be7f7888
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.6282869577407837]], [[0.7110034227371216]], [[0.8070393204689026]], [[0.7731010317802429]], [[0.7212677001953125]], [[0.6343787908554077]], [[0.8950406312942505]], [[0.7803258299827576]], [[0.688999354839325]], [[0.7249688506126404]], [[0.9861186146736145]], [[0.857657253742218]], [[0.720317542552948]], [[0.8511000275611877]], [[0.8740426898002625]], [[0.8391191363334656]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3059ee1302672cfca2a1b326bf8d4a97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81cc2ec7b4f68d2531f14261eea50435
    def get_inputs(self):
        return [
            paddle.uniform([1, 25, 25], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.36853480339050293], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_32cc24b1661c59e627ad7f02138b31ed(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('1'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7a96167cc2a56e818bda1efb1828737b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_32cc24b1661c59e627ad7f02138b31ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d1b4a9252b7cfe14f2801a58375c8b78(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 6, 784, 49], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6c0476e30853a562db7e25b0d555f45e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1b4a9252b7cfe14f2801a58375c8b78
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 784, 49], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.49470216035842896], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ef98933253906ca6f5afcb47de286bb9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3a3435deb3e21c76f9a61b72a31d461
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 3136, 49], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.15890726447105408], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_95fc75089580a6cc63eff68f5e52317f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_81cc2ec7b4f68d2531f14261eea50435
    def get_inputs(self):
        return [
            paddle.uniform([1, 25, 25], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.002016119658946991], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c9d335565e89b2a884c5b85427608c2d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('-0.5'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 80, 80, 2], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_87a44f9439212f268f82353936eda79a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9d335565e89b2a884c5b85427608c2d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 80, 80, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.03263209015130997], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_08f4de02dbb240b69195246a0529dd98(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2, 784, 49], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e292401e8e07d507c5528bcb0bd7f453(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_08f4de02dbb240b69195246a0529dd98
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 784, 49], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2799938917160034], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e3754441f638a9c67bfb6239b281618b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2929d3afd1ef72b0a7674023b6955db5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3754441f638a9c67bfb6239b281618b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.3299209475517273]], [[-0.30752936005592346]], [[-0.2815321683883667]], [[-0.2907193601131439]], [[-0.30475080013275146]], [[-0.32827186584472656]], [[-0.25770995020866394]], [[-0.2887635827064514]], [[-0.31348592042922974]], [[-0.30374887585639954]], [[-0.23305489122867584]], [[-0.2678297460079193]], [[-0.3050079941749573]], [[-0.26960480213165283]], [[-0.26339417695999146]], [[-0.27284806966781616]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.to_tensor([0.14158113300800323], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_61405f4fda6e027f512747343ad1a7cd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 192, 320], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bc4285161ef4d52701b7d64b0d12ca96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61405f4fda6e027f512747343ad1a7cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 192, 320], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2011336237192154], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a7b2a2b08baf91c4ac31d5c5ef087aab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, 256, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_679fa560ca484366ea187638803fca54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7b2a2b08baf91c4ac31d5c5ef087aab
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 256, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.08247071504592896], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0cbbc94b97c3b743d8a945160bc51081(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a67902e8a00a813dace50119c562c00
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 160], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4706328213214874], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_771287239f45bd0110ba0cd04a1e83a8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 64, 48], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_99180e26bf6420da10127937f7dbc1b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_771287239f45bd0110ba0cd04a1e83a8
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 64, 48], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1475658416748047], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c6500aad45cd2d6c4487a485d55ecc16(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 196, 196], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a908e3796ecbf7ad0f9f985599da1a97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c6500aad45cd2d6c4487a485d55ecc16
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 196, 196], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3135160803794861], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_52f5ca14f2e017aa93d03cd9e8186439(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_feced3934b818051896a4e421906a9c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 196, 196], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.27348795533180237], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_394d408aa8bd21a8451cffc136df8fdf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('-0.5'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0439504efb298f8687e72535a642fab7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_394d408aa8bd21a8451cffc136df8fdf
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.6533203125]], [[0.619140625]], [[0.76416015625]], [[0.72021484375]], [[0.7841796875]], [[0.6435546875]], [[0.71533203125]], [[0.67333984375]], [[0.6591796875]], [[0.70361328125]], [[0.6748046875]], [[0.775390625]], [[0.76318359375]], [[0.67919921875]], [[0.7216796875]], [[0.73095703125]]]], dtype='float16').reshape([1, 16, 1, 1]),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_243239ff8282ba0494a5453ba1314329(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 16, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5b1d47ca3e8131f96a8afa25a331ce5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_243239ff8282ba0494a5453ba1314329
    def get_inputs(self):
        return [
            paddle.uniform([196, 4, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.22978518903255463], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ef0d61aca278ee0d4e4c3008032a612d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8400, 2], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f6a4ab069f629226bbce32a38e1539b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef0d61aca278ee0d4e4c3008032a612d
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2540014684200287], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_752f7a9ddcbab4f892a2f3dc46c03d2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f435cf632a2167f6a7b4aceeebcf702d
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 197, 197], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.006860947236418724], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_84e1db2b42dd4da4b460746e3cf5f1e8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 96, 160], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ae8d117d0b20858661ad6bd806e7b3bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84e1db2b42dd4da4b460746e3cf5f1e8
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 96, 160], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2011336237192154], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_038453782e3a2a3b0b673cacf5ffe7ca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_138bc4341418b1873edfa078038640cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_038453782e3a2a3b0b673cacf5ffe7ca
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.14158113300800323], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_74b378d2998a0d15e29511994c738dd7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 144, 32], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4d6606df1ef03b2290280ff5842863b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74b378d2998a0d15e29511994c738dd7
    def get_inputs(self):
        return [
            paddle.uniform([64, 4, 144, 32], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.14657902717590332], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_93fb68f3a3ad31565cadfe27c98cbaec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b5bf36971d16bcec9193fee3f4b80d07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93fb68f3a3ad31565cadfe27c98cbaec
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c4c03b11888c4f443be9ccdd0b79282b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('-0.5'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_88602c42e60ba0e376f36a828a979e0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4c03b11888c4f443be9ccdd0b79282b
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9982350d47e3a6c2f9ae3dcea7e7de8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9efe977f3774294c446c4ffedf6f5752
    def get_inputs(self):
        return [
            paddle.uniform([1, 25, 25], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3368866443634033], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9bc9a17b24ad39ad714dc21f26804579(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 12, 196, 49], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fe725bd7ddc924e7fa8915d31cf9fa53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9bc9a17b24ad39ad714dc21f26804579
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 196, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2404094934463501], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4ffe50d1bc9e7991d68023e3d25ff3b6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('-0.5'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cd21df50087257977a9ab722efc57fd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4ffe50d1bc9e7991d68023e3d25ff3b6
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_424341ec838d2a5dc5e7875571838b20(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('1'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 26, 512], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f6e934d029b0522c9ce47dca32d62e2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_424341ec838d2a5dc5e7875571838b20
    def get_inputs(self):
        return [
            paddle.uniform([1, 26, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3119323253631592], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8f5d8752a777718aef773c02c348e9db(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 26, 256], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_766766d68347c1ad06b4732fa13d8f58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f5d8752a777718aef773c02c348e9db
    def get_inputs(self):
        return [
            paddle.uniform([1, 26, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.38372936844825745], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e90afb7567947a9df1c4d6ed2a8605f2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('1'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_77fb1257dbc73dfff0b97db3a35f8b7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e90afb7567947a9df1c4d6ed2a8605f2
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a5600aab6978f4d254072990ca9cc243(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2125, 2], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_717bea87ba6969871423c00abdc68487(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a5600aab6978f4d254072990ca9cc243
    def get_inputs(self):
        return [
            paddle.uniform([1, 2125, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4863912761211395], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_33ff8707488c7b13a471f451565cb793(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_612d5452deea53476be4c8ebc45a6a45
    def get_inputs(self):
        return [
            paddle.uniform([1, 5, 196, 49], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2799938917160034], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_74002ffaf23d490bd256afcecd9fe14a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 501, 30], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5d854ed32a76ea6c13cc219a9f9d3775(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74002ffaf23d490bd256afcecd9fe14a
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 30], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3747236430644989], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_943a38b40022637350af37fdb1dcd093(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 1024, 2048], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cc652c94c35e67d434a96266e6550926(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_943a38b40022637350af37fdb1dcd093
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 1024, 2048], dtype='int64'), 'int32'),
            paddle.to_tensor([0.19310657680034637], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0edb818bd8daa54ff0d590663719297f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b06b2dae45f5f1e68edcf4150ecc4a11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0edb818bd8daa54ff0d590663719297f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.21432095766067505], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2113a4f958377791d0bb9fc6e80718e0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 12, 49, 49], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_014772a56f1ef26f02ba16b285947643(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2113a4f958377791d0bb9fc6e80718e0
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 12, 49, 49], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.41277825832366943], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_7495b22aaa30431cdac15d8c81c5d4b6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 20, 20, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3460e8261fbb581d6d5f75f13fe0e3a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7495b22aaa30431cdac15d8c81c5d4b6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 20, 20, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2297879308462143], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_7cbd94aa013408246cd07d182c6adb7d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 26, 37], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a71b1becfd9928fc605e75010f3bec9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cbd94aa013408246cd07d182c6adb7d
    def get_inputs(self):
        return [
            paddle.uniform([1, 26, 37], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.42372840642929077], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7b76cc8835be8c59dee83519c1450d47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a67902e8a00a813dace50119c562c00
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 160], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3401828408241272], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9b035973c78d167008f0ba8c3d4f05af(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('1e-06'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 1, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0b6ecb70db477ce0906bc59d3ffdefd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b035973c78d167008f0ba8c3d4f05af
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1, 1024], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.11356471478939056], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2e8e13de325d2ca5790d6947507d01ca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 40, 40, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5fad2cea690f0dcc5a000dab4f204eb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2e8e13de325d2ca5790d6947507d01ca
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 40, 40, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2297879308462143], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_de97861e8227817be68845cbe3488a83(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 350, 25], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b601d4b73f6be18ee27f5ae06932fad4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de97861e8227817be68845cbe3488a83
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 350, 25], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.19558078050613403], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_35e9d3cdfa1baf0a94d6b888647ebfdd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('1'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2979878b1aeea81116d0817710161172(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_35e9d3cdfa1baf0a94d6b888647ebfdd
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c8a8f24f7662572a6b3ac29fe6abb4bf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 12, 49, 49], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_70464bea087dae79d6fd57475d02b86a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8a8f24f7662572a6b3ac29fe6abb4bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 12, 49, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2404094934463501], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_16be7ce6689cecf5cdb90b195663479c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_058d6b68c4b80447dd5048279a61bd5c
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 196, 49], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.49470216035842896], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_126a445fa82743bf5a604d6c94f4742b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3178b76fe216295bf0d228d3788783cc
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 49, 49], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2799938917160034], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_daa99d6529fc5ea0d8a909a843ebc9e2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 144, 64, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_231e5a4f345128bb975e869ef958386e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_daa99d6529fc5ea0d8a909a843ebc9e2
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 128, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.32212451100349426], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_afcf48f4f9040f06b0e6e0d8b9128287(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 24, 49, 49], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c915d23d35e2bcd1ad5494d0c6066f50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_afcf48f4f9040f06b0e6e0d8b9128287
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 24, 49, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2404094934463501], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_3a854f08a8c21143dad11bcb9a87fda1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 144, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_80eafc5550c6346758470c7f19b0fc6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a854f08a8c21143dad11bcb9a87fda1
    def get_inputs(self):
        return [
            paddle.uniform([64, 4, 144, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.27207040786743164], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f544c84469f720f087c95832f36552f7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 25, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_56a178d568b97f7dee7509c41e67fa06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f544c84469f720f087c95832f36552f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.22651496529579163], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a7a8994192b99df8a0d3bb29220ec277(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 12, 577, 577], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cdcba702559320d12cbc176e890bad33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7a8994192b99df8a0d3bb29220ec277
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 577, 577], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4514082372188568], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_aaff8317653e109907318cd4c7b7bcb4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_803f27be4af0272fa285c01710300084(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aaff8317653e109907318cd4c7b7bcb4
    def get_inputs(self):
        return [
            paddle.uniform([8, 256, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.07418585568666458], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cb72da89ac6b1d879dfd93c6680c88cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_943a38b40022637350af37fdb1dcd093
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 1024, 2048], dtype='int64'), 'int32'),
            paddle.to_tensor([0.4215003252029419], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_cd54aef693742d2aa80f30d065d70f92(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 6, 49, 49], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_035b196b627ed532f16db7cf5ac4c789(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd54aef693742d2aa80f30d065d70f92
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 6, 49, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2404094934463501], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_57cb38c5c81c9f7da51c4cdd54023bf9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 320], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d0d029f936ec18918c8737adf4e15e14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57cb38c5c81c9f7da51c4cdd54023bf9
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 320], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3930826485157013], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_da622761fd6501e8e0df773940907292(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 38], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dd42ada6918e976e68721a9ff0a98cd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da622761fd6501e8e0df773940907292
    def get_inputs(self):
        return [
            paddle.uniform([1, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.266884982585907], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b7d8cb9475cc81600d02e9b6c6ae1a55(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 384, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b1988044c5a166b5ef11ae89e9d4158e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7d8cb9475cc81600d02e9b6c6ae1a55
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5ba6eb30eb833543c58a4e88675ff361(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2, 56, 24], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8b73d4295c2347a2a2c779481de7cc35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ba6eb30eb833543c58a4e88675ff361
    def get_inputs(self):
        return [
            paddle.uniform([56, 2, 56, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17822237312793732], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e5d6dd6282cdd571e781b5efc15037b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc84fb4115250adeb7c266147da2c815
    def get_inputs(self):
        return [
            paddle.uniform([1, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.05577254667878151], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ff02ed0f954505ccecada6fc4028dcd3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 256, 36], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1ea7657bef62d1cc2a035f390a2addab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff02ed0f954505ccecada6fc4028dcd3
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 256, 36], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.21719443798065186], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b8b402e7f50f21dc5efb46db79f1924d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, 144, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9cfbb2e60268c5576efce0865cda1fed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8b402e7f50f21dc5efb46db79f1924d
    def get_inputs(self):
        return [
            paddle.uniform([16, 8, 144, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.27207040786743164], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_847f22ce5dfda337575de141972d2dae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('-0.5'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 768, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5db611d5776896277cdbffc737db493c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_847f22ce5dfda337575de141972d2dae
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_dc20095fc856a6819f9b7f63b1531631(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 144, 32], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b1d64d2b900e67b0e262d596cf44d051(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc20095fc856a6819f9b7f63b1531631
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 144, 32], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.14657902717590332], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cc693392e9218ee653f44a8a12b4683b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1029fb76832821d17016697d67418c8a
    def get_inputs(self):
        return [
            paddle.uniform([3200, 20], dtype='float64', min=0, max=0.5),
            paddle.to_tensor([0.2728694677352905], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e158d5761749680f0def38d0a219990c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0edb818bd8daa54ff0d590663719297f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.44748714566230774], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_18840cc130abc3e996ad2cfe6b8af76e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5272a72f3031ccdcf287936b1b4a22ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 40, 40, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4837779998779297], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ef2ff160134e366ed22afc2011e854ff(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 180, 320], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_795203e697e9ac2f17621911d7fce227(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef2ff160134e366ed22afc2011e854ff
    def get_inputs(self):
        return [
            paddle.uniform([1, 180, 320], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2011336237192154], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4e02c4faf498a49d42ce7b13a372d716(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b26901dcc8456298d722a5d85f385871(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e02c4faf498a49d42ce7b13a372d716
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.14158113300800323], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d6d21c4fbec3688dcfb8406f05de45bf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 26, 37], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a7274e93165b6f2d70c861b00311065e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d6d21c4fbec3688dcfb8406f05de45bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 26, 37], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3323487341403961], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_13de917a1b55c3bcdb68df4d7a1b4a57(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, 49, 196], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b36d5adc9e52f82c45769ec1682c5a25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_13de917a1b55c3bcdb68df4d7a1b4a57
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 49, 196], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3135160803794861], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ea706aa1f5843bef3cb1041769a5c0c7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('-0.5'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 20, 20, 2], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1ae51243dfeee26698155f996c4978e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea706aa1f5843bef3cb1041769a5c0c7
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 20, 20, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.03263209015130997], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5864996a4176a8332fdc215177fb798f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('-1'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_99c837f5981668c5d6813e75804b1486(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5864996a4176a8332fdc215177fb798f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2728694677352905], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_fb2898aa0235a53a4785b579e64c4585(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 16, 60], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7d97e5c5005b04cc46a2a32cfbb27b77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb2898aa0235a53a4785b579e64c4585
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 16, 60], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2064722180366516], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d04a41589b9a9dace34e4c630ee54256(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('-0.5'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d677fdde68edbbf71a31431462c62825(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d04a41589b9a9dace34e4c630ee54256
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e2e9a058defbcb25104f4320d3c4bb81(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 12, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5893a0911fd5907112d6ab5983a7021e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e2e9a058defbcb25104f4320d3c4bb81
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.011260986328125]], [[-0.0105133056640625]], [[-0.01250457763671875]], [[-0.0118865966796875]], [[-0.0083770751953125]], [[-0.010894775390625]], [[-0.0131988525390625]], [[-0.00815582275390625]], [[-0.00984954833984375]], [[-0.01152801513671875]], [[-0.01430511474609375]], [[-0.0120086669921875]]]], dtype='float16').reshape([1, 12, 1, 1]),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2f514c54b89a97c5ceaf52f619ff74ea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 64, 48], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3f73db5d7e823ea949cc30c30640eec9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f514c54b89a97c5ceaf52f619ff74ea
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 64, 48], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2749481797218323], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d141b00e2de9272762242c541fd4305f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba4dfdd5d38b89a88e72f81c8afd60c2
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 784, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.446737140417099], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_babb3b026543d60b136647c51111c201(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 72, 180, 320], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2ee48270b13fd307e11fd4fdc1fd64ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_babb3b026543d60b136647c51111c201
    def get_inputs(self):
        return [
            paddle.uniform([1, 72, 180, 320], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1966366469860077], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_24e8bf8d42bbb03413692d0b4c7e8a76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9878f8bb0cb5b468123b76dcd7f188c
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.15963655710220337], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_175089c83cd93dcb84da5f6f8722496a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('-1'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2260cabb650651b21681e443c8488555(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_175089c83cd93dcb84da5f6f8722496a
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1769556850194931], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_acdf505dd4442c38e05c2582fb0984c4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('1'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8fcac09a4c7e04cd362b256fca57dc98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_acdf505dd4442c38e05c2582fb0984c4
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8b3d848c27456fe11d97273d5d97b4cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_caa5c43ac9c047133bcad755afc7155b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 3136, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3837188482284546], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8f2c01afa6b9d48a12e300a00f713a47(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 56, 24], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_60c9ca2d55abac4f356fe26a57e9d7e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f2c01afa6b9d48a12e300a00f713a47
    def get_inputs(self):
        return [
            paddle.uniform([14, 4, 56, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2251664251089096], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_73b2ff87c08e09baff629f0b67b97805(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef2ff160134e366ed22afc2011e854ff
    def get_inputs(self):
        return [
            paddle.uniform([1, 180, 320], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3930826485157013], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_070c77c0943dab6d930441c42684aa81(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 501, 4], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_15be1a865adfb9091eb2245ef22b3de1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_070c77c0943dab6d930441c42684aa81
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 4], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.18577031791210175], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f0d7ae00580d6c34d5c845d6d99abe40(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('1e-06'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 8, 1, 256], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_39c2bb6e86f68f9c8260bbd289a2f988(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0d7ae00580d6c34d5c845d6d99abe40
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 1, 1024], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.35355493426322937], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5fc27c516633e56cd67b9800a694b3a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc84fb4115250adeb7c266147da2c815
    def get_inputs(self):
        return [
            paddle.uniform([1, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4428243041038513], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_916ce35650e898ed97f0ea4b0f9ec8c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e9878f8bb0cb5b468123b76dcd7f188c
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.25278934836387634], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cc09bfc9e0baafeb8349bfc9cc685f0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9959e5c78bca92005cb456ca8ebea07c
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.054565705358982086], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_94a6a1c3a37b42600ee167106a763cc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9e5161e7c922b956949852bb769e231
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.48962774872779846], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_01710257d4c7902c37f80e6c271c88d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1029fb76832821d17016697d67418c8a
    def get_inputs(self):
        return [
            paddle.uniform([3200, 20], dtype='float64', min=0, max=0.5),
            paddle.to_tensor([0.2756720781326294], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0c242caef7b58228088f7c5adf15f578(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc84fb4115250adeb7c266147da2c815
    def get_inputs(self):
        return [
            paddle.uniform([1, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.15963655710220337], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_794011a7517bf9bcc4e2947a08f4e946(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2a8c6021db559dfe726fa0a16fd97d7
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0.47663119435310364], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_bffb7ec75b8e19ed0856dda9e80ae733(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 6], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1892f5c466dc8cfd6e753f88a6825c5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bffb7ec75b8e19ed0856dda9e80ae733
    def get_inputs(self):
        return [
            paddle.uniform([300, 6], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.03263209015130997], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d8fe4804acba4dfd922be54ee7344bf7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 144, 180, 320], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c69bf3429b84572ef9b6dc5b03a5fb08(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d8fe4804acba4dfd922be54ee7344bf7
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 180, 320], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1966366469860077], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2f614c7ecc72775de899a2ec64d4ff87(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 256, 36], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_890b1f37cde8d7015384c9737a72c5a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f614c7ecc72775de899a2ec64d4ff87
    def get_inputs(self):
        return [
            paddle.uniform([4, 4, 256, 36], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.16046729683876038], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_20e31c7ce65552fd73e50734ee396f8c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 26, 70], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_99aae324bd3af7aeff51ae3d44347866(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_20e31c7ce65552fd73e50734ee396f8c
    def get_inputs(self):
        return [
            paddle.uniform([1, 26, 70], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2728694677352905], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4b6865f5174ad0a1cf2e3ca8dcf7e492(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2, 784, 49], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0b1d7e3fa4e82d19083e82832d9d193a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b6865f5174ad0a1cf2e3ca8dcf7e492
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 784, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3837188482284546], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_15c28d6240cf87d2922be1a62cb2993d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d0bf5f24ee0f4c25169b3520aac8ca9
    def get_inputs(self):
        return [
            paddle.uniform([1, 25, 38], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.0033531691879034042], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b624f92c0b7ed2e9bf090c9c8bbd8b04(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 24, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_32535e0286567678e65f7141587910dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b624f92c0b7ed2e9bf090c9c8bbd8b04
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.193359375]], [[-0.1846923828125]], [[-0.2100830078125]], [[-0.2403564453125]], [[-0.1783447265625]], [[-0.192626953125]], [[-0.201171875]], [[-0.228759765625]], [[-0.2032470703125]], [[-0.151123046875]], [[-0.2396240234375]], [[-0.20556640625]], [[-0.2064208984375]], [[-0.192626953125]], [[-0.2291259765625]], [[-0.2178955078125]], [[-0.153564453125]], [[-0.19970703125]], [[-0.241943359375]], [[-0.1494140625]], [[-0.1805419921875]], [[-0.2113037109375]], [[-0.26220703125]], [[-0.2200927734375]]]], dtype='float16').reshape([1, 24, 1, 1]),
            paddle.to_tensor([0.054565705358982086], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_405eea2da36557645581818931ef13ae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('-1'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 192, 320], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3a819b44b3d6c021a66cfc4d28b7e921(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_405eea2da36557645581818931ef13ae
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 320], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.42274439334869385], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_3428b2d63d9b845254794e626c862759(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 36, 180, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2e058d02a936c8eeae596901afbe8f57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3428b2d63d9b845254794e626c862759
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1740708202123642], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_73efb177f6d4bcefc3637eede434febd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('-0.5'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 24, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_11e767167f9ded83c05c86933a3669fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73efb177f6d4bcefc3637eede434febd
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.5918310284614563]], [[0.8131939768791199]], [[0.771870493888855]], [[0.7580124735832214]], [[0.8121208548545837]], [[0.7120105028152466]], [[0.7660691142082214]], [[0.7746148705482483]], [[0.7948158383369446]], [[0.70881587266922]], [[0.7419081330299377]], [[0.6021258234977722]], [[0.713854193687439]], [[0.7082594633102417]], [[0.7725907564163208]], [[0.7512067556381226]], [[0.624729573726654]], [[0.6152797341346741]], [[0.8247078061103821]], [[0.7464378476142883]], [[0.7393931746482849]], [[0.8544216156005859]], [[0.7520461082458496]], [[0.6818184852600098]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_95ce09f878d9682d07776a5a8f89a6e0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 3, 49, 49], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_da4a5e6151a149dcb0253f3a815a5501(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_95ce09f878d9682d07776a5a8f89a6e0
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 3, 49, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2404094934463501], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_90918cad960ee70813d959824225ff8f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('-1'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 180, 320], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f3089ac676a6d17f97a31936c538f160(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_90918cad960ee70813d959824225ff8f
    def get_inputs(self):
        return [
            paddle.uniform([1, 180, 320], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.42274439334869385], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ddcda9e4d70a74de0a32ef3c0de22466(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('-0.5'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1024, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6c87169ac47970d492baa8119ec1567e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ddcda9e4d70a74de0a32ef3c0de22466
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_dec5a73859bae99390e780b2fff0267b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 768, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ecf7a5fec0b0942ba4f645ca96cae998(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dec5a73859bae99390e780b2fff0267b
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.14158113300800323], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5bca407e896b1f055921e7690ae63488(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ee0aadc20b939c5815312a4e8587c52
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 80, 80, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.46604785323143005], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_bec79754fac1f327fc203084510be8f3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('-1'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 48, 80], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_68465ee0e0f69cc247b155077129df41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bec79754fac1f327fc203084510be8f3
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 80], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.42274439334869385], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f30178d20850f0ac58a610e7dcaa978c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 36, 180, 320], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f146279dbb8afc516013e93447c38d19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f30178d20850f0ac58a610e7dcaa978c
    def get_inputs(self):
        return [
            paddle.uniform([1, 36, 180, 320], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.23828814923763275], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_3a1761c721186b570acabac49f9d2192(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_447d319dbe2a1698ecbcbacb95ad4039(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a1761c721186b570acabac49f9d2192
    def get_inputs(self):
        return [
            paddle.uniform([1, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.26706647872924805], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_dcf4ba6c973916932e7826e425017111(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('1'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 26, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0a1bcfdc0986035aeebfdbe738f66ac9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dcf4ba6c973916932e7826e425017111
    def get_inputs(self):
        return [
            paddle.uniform([1, 26, 512], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.26814162731170654], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2b161418f7d777037aa150ccaa88d53d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3178b76fe216295bf0d228d3788783cc
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 49, 49], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3803968131542206], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_93941ddb1a4341cb143f7a4a02eeaac8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 24, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a81f45c394a4d1606b41f3eaa44c0f90(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_93941ddb1a4341cb143f7a4a02eeaac8
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.33978965878486633]], [[-0.2798660695552826]], [[-0.2910524606704712]], [[-0.2948038578033447]], [[-0.28015658259391785]], [[-0.3072567284107208]], [[-0.2926229238510132]], [[-0.2903095483779907]], [[-0.2848410904407501]], [[-0.30812153220176697]], [[-0.2991633713245392]], [[-0.33700284361839294]], [[-0.3067576289176941]], [[-0.3082721531391144]], [[-0.2908574938774109]], [[-0.29664620757102966]], [[-0.3308839201927185]], [[-0.3334420323371887]], [[-0.27674925327301025]], [[-0.29793715476989746]], [[-0.2998441755771637]], [[-0.2687056362628937]], [[-0.2964189946651459]], [[-0.31542980670928955]]]], dtype='float32').reshape([1, 24, 1, 1]),
            paddle.to_tensor([0.14158113300800323], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b288bbf129f068c4577bf8af31e1dc73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1aa503c45f4a343283f38209d9c856fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 80, 80, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.05878882110118866], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_19714ad66c9c575024042c548088bd8a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('-1'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 96, 160], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_aebfd48e2a4d5135b8299cf757c8836d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19714ad66c9c575024042c548088bd8a
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1769556850194931], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8579da1a0be175c4290cba92303844d4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('1'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_439554c6bfb4e9478b767fb9caea12a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8579da1a0be175c4290cba92303844d4
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.0146636962890625]], [[-0.01531982421875]], [[-0.01251983642578125]], [[-0.01336669921875]], [[-0.01213836669921875]], [[-0.01485443115234375]], [[-0.013458251953125]], [[-0.0142822265625]]]], dtype='float16').reshape([1, 8, 1, 1]),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b96870c761db869103e1042b9f143a45(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('-0.5'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3423ca046a78e52f23cec32f1ddebd37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b96870c761db869103e1042b9f143a45
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0a758307a3f21d9f2698d1186e658135(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, 144, 32], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7a7c4e88af0121a8204b6a397ce98810(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a758307a3f21d9f2698d1186e658135
    def get_inputs(self):
        return [
            paddle.uniform([16, 8, 144, 32], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.14657902717590332], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6f6b6d9ac69ab145ab1e506636e7b739(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 96, 160], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_497371ee844c74b4c60703f2d2f384b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f6b6d9ac69ab145ab1e506636e7b739
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 96, 160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4670775532722473], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2ee1ab50853226bc090c557dfe293706(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7495b22aaa30431cdac15d8c81c5d4b6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 20, 20, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2573584020137787], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a4e7fad10f43a1098b026b503311f396(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 196, 196], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7d08066524fbd40562b12e4b102be39c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4e7fad10f43a1098b026b503311f396
    def get_inputs(self):
        return [
            paddle.uniform([1, 4, 196, 196], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.13168077170848846], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c8498ed019c0f3889552626e6a1bdcaa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('-0.5'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 24, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_544bcb0244e3d3951af859fa004ed149(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8498ed019c0f3889552626e6a1bdcaa
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[0.8662109375]], [[0.890625]], [[0.81884765625]], [[0.7333984375]], [[0.90869140625]], [[0.8681640625]], [[0.84423828125]], [[0.76611328125]], [[0.83837890625]], [[0.9853515625]], [[0.7353515625]], [[0.83154296875]], [[0.8291015625]], [[0.8681640625]], [[0.76513671875]], [[0.796875]], [[0.978515625]], [[0.84814453125]], [[0.72900390625]], [[0.990234375]], [[0.90234375]], [[0.8154296875]], [[0.67138671875]], [[0.79052734375]]]], dtype='float16').reshape([1, 24, 1, 1]),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_84a676afbe1c971951a5d2f485410691(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a8db6d7dca929d2b3e25f2941011704
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4670775532722473], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5c13af7b318da3d062b1f2e5cdc81cb2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_922c9842dec687687567dafea6d65cd7
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 49, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.016803564503788948], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e39b71e32c55dc4457c232c913b772ac(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_45d05bda12878a2bd4b46ea0d40d85a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e39b71e32c55dc4457c232c913b772ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_7f6099b690f18c1fd56de23b6738399b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4211ed1bb683465244e909ddec846676(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f6099b690f18c1fd56de23b6738399b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.13788749277591705], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_3933d070df53e7219873e3c546dc7bb1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('1'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_da5792513fb6cc49f3f6ac63c1b5f408(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3933d070df53e7219873e3c546dc7bb1
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.01204681396484375]], [[-0.007965087890625]], [[-0.01189422607421875]], [[-0.007965087890625]], [[-0.007965087890625]], [[-0.01058197021484375]], [[-0.01175689697265625]], [[-0.01242828369140625]], [[-0.007965087890625]], [[-0.01273345947265625]], [[-0.00946807861328125]], [[-0.008758544921875]], [[-0.01068115234375]], [[-0.01027679443359375]], [[-0.00921630859375]], [[-0.00937652587890625]]]], dtype='float16').reshape([1, 16, 1, 1]),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_aaaa401a33574f0718be12ff659a34d5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 20, 20, 2], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_66cb84ff68f314cc81077f5bb5ebb6fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aaaa401a33574f0718be12ff659a34d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 20, 20, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.13843213021755219], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_28944b3ff4fd499821d448ea0de1dff6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1029fb76832821d17016697d67418c8a
    def get_inputs(self):
        return [
            paddle.uniform([3200, 20], dtype='float64', min=0, max=0.5),
            paddle.to_tensor([0.266884982585907], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ed8d9b5769e915b3f1bb25366741ddef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('1'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4390a94d6143e69aee90d9574c72f3c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ed8d9b5769e915b3f1bb25366741ddef
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d93b9456604063f4eed57be1858da906(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('-0.5'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 80, 80, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_24d4cc66821316882a57ac0596245ead(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d93b9456604063f4eed57be1858da906
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 80, 80, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.22985711693763733], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_638deaf8343289ae46f5545f215dd5b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_08f4de02dbb240b69195246a0529dd98
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 784, 49], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.15890726447105408], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_3fc7ed85e88d23919088a295d0c39261(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6562277230f2d4cc77121fabe27705ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3fc7ed85e88d23919088a295d0c39261
    def get_inputs(self):
        return [
            paddle.uniform([1, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.009423283860087395], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_453571dae34964880d201236aa592b45(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 49, 24], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8df89fd7ce7402ab712ad6bc14032403(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_453571dae34964880d201236aa592b45
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 49, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2251664251089096], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6764fc8329509a1fc0eab47aeabd3784(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 64, 350, 25], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1b75f954f733348097613ddee1961c31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6764fc8329509a1fc0eab47aeabd3784
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 350, 25], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.07456143200397491], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8230b0ebeeeeb7bb6ba41ba6393b9991(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2e8e13de325d2ca5790d6947507d01ca
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 40, 40, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.05878882110118866], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_bd17fed2ac559e538d0868702abc96be(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a3259a6fd4d084f1349814ffc8f959bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bd17fed2ac559e538d0868702abc96be
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_598ba4ba7a1e1b66eb61449bb4c6c74f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 501, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e83f6545238571999417e6a60335e9b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_598ba4ba7a1e1b66eb61449bb4c6c74f
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.12629848718643188], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_17ca862fbd4e91e23f2aadb3f7fb9daa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 144, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_25a65baef8c7aec3557a038ff6f69156(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_17ca862fbd4e91e23f2aadb3f7fb9daa
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 144, 32], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.27207040786743164], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_372443436f07a33041fe1f6229e8ea63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57cb38c5c81c9f7da51c4cdd54023bf9
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 320], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3062542974948883], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6a993c71471112fcb702ddcffd46aac6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9bc9a17b24ad39ad714dc21f26804579
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 196, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.446737140417099], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b39345475ccb943515ee22ea43a38aa9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 24, 49, 49], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_efaeddc657c037666bc418fe517def53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b39345475ccb943515ee22ea43a38aa9
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 49, 49], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.49470216035842896], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b6245ef7cf3574a6f0fa43d39b57203e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57cb38c5c81c9f7da51c4cdd54023bf9
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 320], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2011336237192154], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e44f0f6fee3d1ac3da9e520a6a8d28a2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 3136, 49], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ef21229ccb65849836d6864ce7c4991f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e44f0f6fee3d1ac3da9e520a6a8d28a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 3136, 49], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.41277825832366943], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eb6824cc78e63ecb648e2aadc950971f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9e5161e7c922b956949852bb769e231
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.06225132569670677], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c4914b03d84f4bb135c58476b03dedb7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a8db6d7dca929d2b3e25f2941011704
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.49733033776283264], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_687c7f27f8769ec680d8e81172cbafcf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('-1'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b556c580da21dda959f25bcaa04a8436(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_687c7f27f8769ec680d8e81172cbafcf
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 100], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0021241477224975824], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c0442243bf155553e3ddbf87c8bc2a72(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('1'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 12, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_90b7c4155cef501cfc8e9647873fddd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c0442243bf155553e3ddbf87c8bc2a72
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.01055145263671875]], [[-0.01007843017578125]], [[-0.01146697998046875]], [[-0.01311492919921875]], [[-0.009735107421875]], [[-0.0105133056640625]], [[-0.01097869873046875]], [[-0.012481689453125]], [[-0.0110931396484375]], [[-0.00824737548828125]], [[-0.0130767822265625]], [[-0.0112152099609375]]]], dtype='float16').reshape([1, 12, 1, 1]),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1a804acf86794ca40e70d8b1b3ad0bc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9e5161e7c922b956949852bb769e231
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.15963655710220337], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d280d74cd849e3ec5602b0906cea7302(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a4e67efd285f59392666614cdcb6095b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d280d74cd849e3ec5602b0906cea7302
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6d6b3b16c1b032daf5bd02d2e6617792(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 32, 180, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7a969b3e5f836fe4e870b14ce8e39637(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6d6b3b16c1b032daf5bd02d2e6617792
    def get_inputs(self):
        return [
            paddle.uniform([2, 32, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1769556850194931], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6470578993141845927a23a2b566c321(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 49, 24], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8319c1edeb88abe96bcb318cf292fa4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6470578993141845927a23a2b566c321
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 49, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.17822237312793732], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0057168b90d3f7dbb951c702a0d4ed80(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3591d0a242cb9eedaf11a1b1b7114ce
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.14158113300800323], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bad2affec343d864c06fd4bb2072da66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1aa503c45f4a343283f38209d9c856fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 80, 80, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2297879308462143], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e1018bd111dafabc09b32c7a9b33cb7f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc84fb4115250adeb7c266147da2c815
    def get_inputs(self):
        return [
            paddle.uniform([1, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.16184796392917633], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_504dfaac90192278dada2704bb8a4fff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9e5161e7c922b956949852bb769e231
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.345152884721756], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9f44d28ac508294c2c619f47b950e30a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_91042b5ffb6a1bc5f552b011eb6ae031(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f44d28ac508294c2c619f47b950e30a
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.01454925537109375]], [[-0.01369476318359375]], [[-0.01424407958984375]], [[-0.01230621337890625]], [[-0.012542724609375]], [[-0.01416015625]], [[-0.01334381103515625]], [[-0.01316070556640625]]]], dtype='float16').reshape([1, 8, 1, 1]),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_701c46be5f6947a729e23f0508d60bf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_175089c83cd93dcb84da5f6f8722496a
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.21931400895118713], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ac29c1418460cff71a71e343d1db771c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef2ff160134e366ed22afc2011e854ff
    def get_inputs(self):
        return [
            paddle.uniform([1, 180, 320], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.0881076231598854], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ab904c30132e9e5b4717dc6e9f67b4d1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('-1'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 48, 80], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f585f05d799fb67e6dab52cf004253c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ab904c30132e9e5b4717dc6e9f67b4d1
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 80], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1769556850194931], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6a65ce2202b50e7387a79f446ff67556(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc84fb4115250adeb7c266147da2c815
    def get_inputs(self):
        return [
            paddle.uniform([1, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.21931400895118713], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_81bc1380cce31ebce7059ab0bde39dd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b3eb00949617b37c0f23b6f80426c373
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.14158113300800323], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_fc416e6000a7a7fc1f2fea4998904495(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('-0.5'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 40, 40, 2], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_507533464226fda1b3d7b35567af6c32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc416e6000a7a7fc1f2fea4998904495
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 40, 40, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.03263209015130997], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_51c6c801f5aa02f4b87bc41f372ca4d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_19714ad66c9c575024042c548088bd8a
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.21931400895118713], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ad9862f20df89c280ab9422dfac2166a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('-0.5'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 40, 40, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_064b47629b4f8e8c8a4ceeab14abae78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ad9862f20df89c280ab9422dfac2166a
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 40, 40, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.22985711693763733], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_910e7f7c05bf994a47cf978d96c7b500(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7f6099b690f18c1fd56de23b6738399b
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2763035297393799], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9327538286d2df5c994b659dd3fec4e2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8b8e4f7b13a35aa2cc297c5173c478f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9327538286d2df5c994b659dd3fec4e2
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.0120391845703125]], [[-0.01102447509765625]], [[-0.009490966796875]], [[-0.00812530517578125]], [[-0.01200103759765625]], [[-0.007965087890625]], [[-0.0115509033203125]], [[-0.009307861328125]], [[-0.007965087890625]], [[-0.0091705322265625]], [[-0.00826263427734375]], [[-0.01119232177734375]], [[-0.00855255126953125]], [[-0.00807952880859375]], [[-0.008056640625]], [[-0.007965087890625]]]], dtype='float16').reshape([1, 16, 1, 1]),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5c5d67c4348741a7beacf942dab530c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3dd3e2336f9e1ae1d0d594882c5f5f88
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.054565705358982086], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_459323979d7b97cf18a7ae2e933a4753(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 26, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_aeaab8b4acaa67129f74e06a5c008f0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_459323979d7b97cf18a7ae2e933a4753
    def get_inputs(self):
        return [
            paddle.uniform([1, 26, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.02612169273197651], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9c188bba9953459cb17855bfbec67096(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 384, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d22a35cdd556f81e1717cff66f8c0bac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9c188bba9953459cb17855bfbec67096
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2d89ef761e870396170a7ab3f5c75522(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2e8e13de325d2ca5790d6947507d01ca
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 40, 40, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.39329299330711365], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_75a53a82fa2dce5e2f89773bbafbc9e9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 26, 70], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_47b29cc5ee63145bf07853b53edb6799(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75a53a82fa2dce5e2f89773bbafbc9e9
    def get_inputs(self):
        return [
            paddle.uniform([1, 26, 70], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0021241477224975824], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ccda5dc9791141d0735235975db9fe4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_409e5e4b343cf4ccc512e0686e2e1f03
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 49, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2404094934463501], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2fc3e6663d588f3fa40589eb25eb9603(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('-0.5'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b036354e31f0a1f66538ca0151e5d6c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2fc3e6663d588f3fa40589eb25eb9603
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6a3a0333a44dbe2440e432560f7a8d79(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 12, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7c8a1d5f85477c84ed10bb0cfc001223(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a3a0333a44dbe2440e432560f7a8d79
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.04343109205365181]], [[-0.04364551976323128]], [[-0.04117993265390396]], [[-0.04199950769543648]], [[-0.046846918761730194]], [[-0.04720909893512726]], [[-0.039182472974061966]], [[-0.042182281613349915]], [[-0.04245227947831154]], [[-0.03804364800453186]], [[-0.04196733608841896]], [[-0.04465891048312187]]]], dtype='float32').reshape([1, 12, 1, 1]),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_839005e6aa2601ae66fe13796f4e36c7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, 49, 196], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_08f999b9502752d4b7197e1eec047a39(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_839005e6aa2601ae66fe13796f4e36c7
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 49, 196], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.13168077170848846], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ef91b6028221ebbce68331b4808d5b6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_519af1151b8858cafe172b6684ee2310
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 192, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.15963655710220337], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_66d26275c2546482adbe2d2e6f18db2b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('-0.5'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dc97e96c240e5d221ca3e9e0db76410b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_66d26275c2546482adbe2d2e6f18db2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3992eb503a984a3e5b96ae334b47d433(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_74002ffaf23d490bd256afcecd9fe14a
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 30], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.12629848718643188], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_32775831ce4bd45aac1ccb5cc51b7b89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_070c77c0943dab6d930441c42684aa81
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 4], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.37192127108573914], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_434d8c8ee00e7c3c86af4caf669272c8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 768, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_36bfaae6d106076e5f90d7ba8343efb1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_434d8c8ee00e7c3c86af4caf669272c8
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.054565705358982086], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b7b291ec26c85f0a0e95d3575e6cd28a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('-0.5'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1024, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d96a9cca89295b08c5fff1f58b70d1c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b7b291ec26c85f0a0e95d3575e6cd28a
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_7412450ac6a067c6b75c318dd6e1b64c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 6, 49, 49], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_37a6b8ed9a10f3c38ab968819be2d575(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7412450ac6a067c6b75c318dd6e1b64c
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 6, 49, 49], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.41277825832366943], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e8322e2a0e03f8fef3332528c8935a16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b6865f5174ad0a1cf2e3ca8dcf7e492
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 784, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.016803564503788948], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_716bedeff3544940261df4f7dfaf53bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aaaa401a33574f0718be12ff659a34d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 20, 20, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.15257292985916138], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cb2d5ee73a8ea3600e2bbfe60182b9a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d9e5161e7c922b956949852bb769e231
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.31343406438827515], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0b18148018500aab3547c3488b02611a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5272a72f3031ccdcf287936b1b4a22ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 40, 40, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.13843213021755219], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_713e733814e0a71d22ebdc22bbe1a79b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('1'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3ef0ae54ce347f38c72ee5d98e5b1d38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_713e733814e0a71d22ebdc22bbe1a79b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.03653030842542648]], [[-0.03514935448765755]], [[-0.04353402554988861]], [[-0.039983395487070084]], [[-0.037438008934259415]], [[-0.04565148428082466]], [[-0.04420868679881096]], [[-0.04190097376704216]], [[-0.03843735158443451]], [[-0.037291187793016434]], [[-0.041884440928697586]], [[-0.03599511831998825]], [[-0.03529391810297966]], [[-0.040505290031433105]], [[-0.04184701666235924]], [[-0.04465116932988167]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dcc838c61a8b1adb4f30362ca0283639(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9327538286d2df5c994b659dd3fec4e2
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.268798828125]], [[-0.28076171875]], [[-0.2294921875]], [[-0.2449951171875]], [[-0.222412109375]], [[-0.272216796875]], [[-0.2467041015625]], [[-0.26171875]], [[-0.2666015625]], [[-0.2509765625]], [[-0.260986328125]], [[-0.2254638671875]], [[-0.2298583984375]], [[-0.259521484375]], [[-0.2445068359375]], [[-0.2412109375]]]], dtype='float16').reshape([1, 16, 1, 1]),
            paddle.to_tensor([0.054565705358982086], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_09746589edb9763af0309d3e4b293020(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc84fb4115250adeb7c266147da2c815
    def get_inputs(self):
        return [
            paddle.uniform([1, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1769556850194931], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a1de1984348915529b1639b85b2401a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5e40a6343ca3d9d45336185e5ebfbc3
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 80], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.28113576769828796], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3946bb7d4ef8e6a5e0111807b5b0fd84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f544c84469f720f087c95832f36552f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 25, 38], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.266884982585907], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9ffa93de8475bb20ec1e37bbb771eb76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aaaa401a33574f0718be12ff659a34d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 20, 20, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.46604785323143005], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5efbb911aca96f91b42904cf4b30db3d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 24, 49, 49], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1f0561239e6358e2c69538d2712c3e91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5efbb911aca96f91b42904cf4b30db3d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 24, 49, 49], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.41277825832366943], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_af35b634e9fd7a1f1225a6ac8284eac7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 6], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_47902ac18315d397902d2189cb61be85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af35b634e9fd7a1f1225a6ac8284eac7
    def get_inputs(self):
        return [
            paddle.uniform([300, 6], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.22985711693763733], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c8211152ebbf4e7ca46d42b2738a686a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5e40a6343ca3d9d45336185e5ebfbc3
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 80], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.09170089662075043], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f89b3106eede327db976d288ed9bb1e9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('1'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_61ba294a381a491118ae5bce58cdc98d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f89b3106eede327db976d288ed9bb1e9
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.046710580587387085]], [[-0.0435403548181057]], [[-0.03985964506864548]], [[-0.041160374879837036]], [[-0.04314696416258812]], [[-0.04647710174322128]], [[-0.03648686781525612]], [[-0.04088347405195236]]]], dtype='float32').reshape([1, 8, 1, 1]),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b5437e795feec71bb396ac4f3d5eb0e6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('1'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 384, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1b159070150c0ded27c191faa1c04917(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b5437e795feec71bb396ac4f3d5eb0e6
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3540290296077728], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_aaf4b3726387ee0d41c4034f6c091acf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8400, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6a5cb195c9602f59a62d6266cfae5180(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aaf4b3726387ee0d41c4034f6c091acf
    def get_inputs(self):
        return [
            paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.01980632171034813], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5e684ce4279463ff3c8cac05acf810a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_871556bbbd3fa4e4bf71570ffb10d4e4
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 30], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.37192127108573914], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_59888e8a852d86f6385d49801e5fe884(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8a8db6d7dca929d2b3e25f2941011704
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.16694828867912292], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d6f5dc3c9502dd2cc1a26698e4dfb91b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef2ff160134e366ed22afc2011e854ff
    def get_inputs(self):
        return [
            paddle.uniform([1, 180, 320], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.42274439334869385], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a39d3cb7fade49da89c902551eb2a907(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e5560a1dc97ae0d931c3618304f8d503(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a39d3cb7fade49da89c902551eb2a907
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.0443836934864521]], [[-0.04300510883331299]], [[-0.032996173948049545]], [[-0.03791964054107666]], [[-0.043183378875255585]], [[-0.0381709523499012]], [[-0.037291646003723145]], [[-0.03863013908267021]]]], dtype='float32').reshape([1, 8, 1, 1]),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1743383ce1c7a3607877f24b09f205d0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('1'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 32, 100], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_425f31f681f7b4a145ff95653aa0973b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1743383ce1c7a3607877f24b09f205d0
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 100], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.2728694677352905], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0d8ce81f331f907c65a856262735052a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2125, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c139f5b46c8158b0996b10fdeb30ddab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0d8ce81f331f907c65a856262735052a
    def get_inputs(self):
        return [
            paddle.uniform([1, 2125, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.43221747875213623], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d2a0ac3c0c248924b240d29ea1aea55d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2, 32, 180, 320], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_70fa709d933908725cfe5c57738e779e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2a0ac3c0c248924b240d29ea1aea55d
    def get_inputs(self):
        return [
            paddle.uniform([2, 32, 180, 320], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.42274439334869385], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b25a861a9ef845cd76f973d36775f66a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f942792004635b32d9f26fdf066c0696
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.054565705358982086], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9b2636599a6ad0f11d57bdb3b9bd58ed(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1024, 1, 1], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f444c2eda5d4be861290270988cf94e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9b2636599a6ad0f11d57bdb3b9bd58ed
    def get_inputs(self):
        return [
            paddle.uniform([1, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.054565705358982086], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c95d7df9301dae7a01cc4cbe2cf65b71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1b4a9252b7cfe14f2801a58375c8b78
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 784, 49], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.41277825832366943], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_df23fc974e59e08136c5d535c9f68b38(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 144, 64, 64], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b5058413667c8b8619cc2368fa0a93d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df23fc974e59e08136c5d535c9f68b38
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 128, 128], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.15648093819618225], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_74179b49c4929a00a00e7b30c7e91238(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7495b22aaa30431cdac15d8c81c5d4b6
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 20, 20, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.05878882110118866], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_add805d104b64d43f1a39ebe4659dcbc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('-0.5'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 20, 20, 2], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0bb74bf313539a1fa9da37f9c0ff2b1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_add805d104b64d43f1a39ebe4659dcbc
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 20, 20, 2], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.22985711693763733], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_641c6946fa161bb55340356291b944c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ef2ff160134e366ed22afc2011e854ff
    def get_inputs(self):
        return [
            paddle.uniform([1, 180, 320], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4322897791862488], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_63cc20aac66584f34ea6fe3c1f30ca99(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3, 3136, 49], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_40b146dbd8c97c1a500c7dd3cf6af52f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63cc20aac66584f34ea6fe3c1f30ca99
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 3136, 49], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2404094934463501], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5456beccdce0111e9c1c6b92478d04a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e39b71e32c55dc4457c232c913b772ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.14158113300800323], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_14e8756b189ecfb5fb14d31df97d0112(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('0'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, 512], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f05f6cc10912411dc9cdf0564986acb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_14e8756b189ecfb5fb14d31df97d0112
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 512], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3962007164955139], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1ab9c8f7c7c63c74a981e1595acc4728(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('-0.5'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2d4318cffb80490df8ad5719de24311e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ab9c8f7c7c63c74a981e1595acc4728
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7c5f1003614ccb444bc9872396c3a23d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b39345475ccb943515ee22ea43a38aa9
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 49, 49], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.41277825832366943], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_480c8a5d1fbf3fa6d4122ba530f10456(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.scale_(input_0, input_1, float('1'), True)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1db4950b2465310981f534a37b7a4425(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_480c8a5d1fbf3fa6d4122ba530f10456
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_58d55a14b78bd11d4ceef9577b765366(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6f6b6d9ac69ab145ab1e506636e7b739
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 96, 160], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.15963655710220337], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4a3cf92f0b91b4c95bd827fe1245468f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3754441f638a9c67bfb6239b281618b
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[-0.039256930351257324]], [[-0.04020240902900696]], [[-0.04218309000134468]], [[-0.038093797862529755]], [[-0.03595396876335144]], [[-0.03911122307181358]], [[-0.03731000795960426]], [[-0.04295990243554115]], [[-0.040298208594322205]], [[-0.046923860907554626]], [[-0.04241222143173218]], [[-0.034252773970365524]], [[-0.04272951930761337]], [[-0.044259000569581985]], [[-0.03523772582411766]], [[-0.03425834700465202]]]], dtype='float32').reshape([1, 16, 1, 1]),
            paddle.to_tensor([0.2707028388977051], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_88446ced309365b62c5fb260d3d8a4a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1029fb76832821d17016697d67418c8a
    def get_inputs(self):
        return [
            paddle.uniform([3200, 20], dtype='float64', min=0, max=0.5),
            paddle.to_tensor([0.0021241477224975824], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4d40a6b7a758ebf4bf9daeb0e8f49980(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ee0aadc20b939c5815312a4e8587c52
    def get_inputs(self):
        return [
            paddle.uniform([1, 3, 80, 80, 2], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.13843213021755219], dtype='float32').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9bd3afd4d5c6ee3ca4d53b7b694f01fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_598ba4ba7a1e1b66eb61449bb4c6c74f
    def get_inputs(self):
        return [
            paddle.uniform([1, 501, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3747236430644989], dtype='float32').reshape([1]),
        ]


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